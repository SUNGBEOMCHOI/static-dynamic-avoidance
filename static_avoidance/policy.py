import torch
import torch.nn as nn
import logging
from static_avoidance.utils.action import ActionXY, ActionRot
import itertools
import numpy as np
from static_avoidance.utils.state import FullState, SmallState, ObstacleState
import math
import time
import copy

def mlp(input_dim, mlp_dims, last_relu=False):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        layers.append(nn.Tanh())
    layers.append(nn.Flatten(0, -1))
    layers.append(nn.Linear(30, 100))
    layers.append(nn.Tanh())
    layers.append(nn.Linear(100, 50))
    layers.append(nn.Tanh())
    layers.append(nn.Linear(50, 1))
    net = nn.Sequential(*layers).to(torch.device('cpu'))
    return net



class ValueNetwork(nn.Module):
    def __init__(self, input_dim, mlp_dims):
        super().__init__()
        self.value_network = mlp(input_dim, mlp_dims)

    def forward(self, state):
        value = self.value_network(state)
        return value

class SOA():
    def __init__(self):
        self.phase = None
        self.model = None
        self.device = None
        self.last_state = None
        self.time_step = None
        # if agent is assumed to know the dynamics of real world
        self.env = None
        self.kinematics = None
        self.epsilon = None
        self.gamma = None
        self.sampling = None
        self.speed_samples = None
        self.rotation_samples = None
        self.query_env = None
        self.action_space = None
        self.epsilon = None
        self.speeds = None
        self.rotations = None
        self.action_values = None
        self.self_state_dim = 3
        # vx, vy, radius of robot
        self.obstacle_dim = 3
        # px, py between robot and obstacle, radius of obstacle
        self.joint_state_dim = self.self_state_dim + self.obstacle_dim
        self.goal_radius = 0.2

    def configure(self, config):
        self.set_common_parameters(config)
        mlp_dims = [int(x) for x in config.get('SOA', 'mlp_dims').split(', ')]
        self.model = ValueNetwork(self.joint_state_dim, mlp_dims)


    def set_common_parameters(self, config):
        self.gamma = config.getfloat('rl', 'gamma')
        self.kinematics = config.get('action_space', 'kinematics')
        self.sampling = config.get('action_space', 'sampling')
        self.speed_samples = config.getint('action_space', 'speed_samples')
        self.rotation_samples = config.getint('action_space', 'rotation_samples')
        self.query_env = config.getboolean('action_space', 'query_env')

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def set_phase(self, phase):
        self.phase = phase

    def get_model(self):
        return self.model

    def set_device(self, device):
        self.device = device

    def set_env(self, env):
        self.env = env

    def predict(self, state):
        #a = time.time()
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')

        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        current_v = math.sqrt(state.self_state.vx ** 2 + state.self_state.vy ** 2)
        #print(state.self_state.px, state.self_state.py)
        self.build_action_space(state.self_state.v_pref, current_v)
        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            max_action = self.action_space[np.random.choice(len(self.action_space))]
            max_value = '1234'
        else:
            self.action_values = list()
            max_value = float('-inf')
            max_action = None
            goal_state = ObstacleState(state.self_state.gx, state.self_state.gy, self.goal_radius)
            for action in self.action_space:
                next_self_state = self.propagate(state.self_state, action) #로봇의 FullState를 반환
                reward = self.compute_reward(next_self_state, state.obstacle_states)
                if reward == -0.025: #다다음 상황이 충돌일 경우에는 현재 측정하는 action을 버리고 새로운 action을 탐색
                    continue
                next_small_self_state = SmallState(next_self_state.vx, next_self_state.vy, next_self_state.radius)

                # VALUE UPDATE
                tensors = []
                tensors.append(self.input_state(next_self_state, goal_state))

                for i, obstacle_state in enumerate(state.obstacle_states):
                    if not isinstance(next_small_self_state, SmallState) or not isinstance(obstacle_state, ObstacleState):
                        print('인스턴스 에러')
                    tensors.append(self.input_state(next_self_state, obstacle_state))
                    if len(tensors) == 6:
                        break

                while len(tensors) != 6:
                    tensors.append([0.0,0.0,0.0,0.0,0.0,0.0])
                next_state_value = self.model(torch.Tensor(tensors).to(self.device)).data.item()

                value = reward + pow(self.gamma, self.time_step * state.self_state.v_pref) * next_state_value
                self.action_values.append(value)
                if value > max_value:
                    max_value = value
                    max_action = action

        if max_action is None:
            max_action = self.action_space[np.random.choice(len(self.action_space))]
            max_value = '4321'
            #raise ValueError('Value network is not well trained. ')
        if self.phase == 'train':
            self.last_state = self.transform(state).to(self.device)
        #print(time.time() - a)
        return max_action

    def transform(self, all_state):
        RobotState = all_state.self_state
        ObState = all_state.obstacle_states
        GoalState = ObstacleState(RobotState.gx, RobotState.gy, self.goal_radius)
        tensors = []
        tensors.append(self.input_state(RobotState, GoalState))
        for obs in ObState:
            tensors.append(self.input_state(RobotState, obs))
        while len(tensors) != 6:
            tensors.append([0, 0, 0, 0, 0, 0])
        transform_tensor = torch.Tensor(tensors)
        return transform_tensor

    def input_state(self, robot_state, ob_state): #로봇의 풀 스테이트와 장애물의 스테이트를 넣으면 ex, ey, vx, vy, rR, ro를 반환
        ex = ob_state.px - robot_state.px
        ey = ob_state.py - robot_state.py
        return [ex, ey, robot_state.vx, robot_state.vy, robot_state.radius, ob_state.radius]

    def compute_reward(self, robot_full_state, obstacles):
        # collision detection
        dmin = float('inf')
        collision = False
        for i, obstacle in enumerate(obstacles):
            dist = np.linalg.norm((robot_full_state.px - obstacle.px, robot_full_state.py - obstacle.py)) - robot_full_state.radius - obstacle.radius
            if dist < 0:
                collision = True
                break
            if dist < dmin:
                dmin = dist

        # check if reaching the goal
        reaching_goal = np.linalg.norm((robot_full_state.px - robot_full_state.gx, robot_full_state.py - robot_full_state.gy)) < robot_full_state.radius + self.goal_radius
        if collision:
            reward = -0.025
        elif reaching_goal:
            reward = 1
        elif dmin < 0.05:
            reward = (dmin - 0.1) * 0.02 * self.time_step
        else:
            reward = 0
        return reward

    def reach_destination(self, state):
        self_state = state.self_state
        if np.linalg.norm((self_state.py - self_state.gy, self_state.px - self_state.gx)) < self_state.radius+self.goal_radius:
            return True
        else:
            return False

    def build_action_space(self, v_pref, current_velocity):
        """
        Action space consists of 25 uniformly sampled actions in permitted range and 25 randomly sampled actions.
        """
        holonomic = True if self.kinematics == 'holonomic' else False
        speeds = [current_velocity + (v_pref / 10) * (i - 2) for i in range(self.speed_samples)]

        speed_zero = []
        for i in range(len(speeds)):
            if speeds[i] > v_pref:
                speeds[i] = v_pref
            elif speeds[i] <= 0:
                speed_zero.append(i)
        speedss = copy.deepcopy(speeds)
        speeds = []
        for kk in range(len(speedss)):
            if kk in speed_zero:
                continue
            else:
                speeds.append(speedss[kk])


        if holonomic:
            rotations = np.linspace(0, 2 * np.pi, self.rotation_samples, endpoint=False)
        else:
            rotations = np.linspace(-np.pi / 8, np.pi / 8, self.rotation_samples)


        #action_space = [ActionXY(0, 0) if holonomic else ActionRot(0, 0)]
        action_space = []
        for rotation, speed in itertools.product(rotations, speeds):
            if holonomic:
                action_space.append(ActionXY(speed * np.cos(rotation), speed * np.sin(rotation)))
            else:
                action_space.append(ActionRot(speed, rotation))
        try:
            action_space.remove(ActionRot(0, 0))
        except:
            pass
        self.speeds = speeds
        self.rotations = rotations
        self.action_space = action_space

    def propagate(self, state, action):
        if self.kinematics == 'holonomic':
            next_px = state.px + action.vx * self.time_step
            next_py = state.py + action.vy * self.time_step
            next_state = FullState(next_px, next_py, action.vx, action.vy, state.radius,
                                   state.gx, state.gy, state.v_pref, state.theta)
        else:
            next_theta = state.theta + action.r
            next_vx = action.v * np.cos(next_theta)
            next_vy = action.v * np.sin(next_theta)
            next_px = state.px + next_vx * self.time_step * 2
            next_py = state.py + next_vy * self.time_step * 2
            next_state = FullState(next_px, next_py, next_vx, next_vy, state.radius, state.gx, state.gy,
                                   state.v_pref, next_theta)
        return next_state