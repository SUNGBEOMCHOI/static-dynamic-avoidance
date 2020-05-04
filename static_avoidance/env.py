import logging
import matplotlib.lines as mlines
import numpy as np
from matplotlib import patches
from numpy.linalg import norm
import random
from utils.utils import point_to_segment_dist
from utils.obstacle import Obstacle
from utils.info import *
import time
import math

class Env():
    def __init__(self):
        self.time_limit = None
        self.time_step = None
        self.robot = None
        self.obstacles = None
        self.global_time = None
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None
        self.square_width = None
        self.circle_radius = None
        self.obstacle_num = None
        self.states = None
        self.action_values = None
        self.case_capacity = None
        self.case_size = None
        self.grid_size = None
        self.grid_count = None
        self.goal_radius = 0.2


    def configure(self, config):
        self.config = config
        self.time_limit = config.getint('env', 'time_limit')
        self.time_step = config.getfloat('env', 'time_step')
        self.success_reward = config.getfloat('reward', 'success_reward')
        self.collision_penalty = config.getfloat('reward', 'collision_penalty')
        self.discomfort_dist = config.getfloat('reward', 'discomfort_dist')
        self.discomfort_penalty_factor = config.getfloat('reward', 'discomfort_penalty_factor')

        self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
        self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': config.getint('env', 'val_size'),
                          'test': config.getint('env', 'test_size')}
        self.square_width = config.getfloat('sim', 'square_width')
        self.circle_radius = config.getfloat('sim', 'circle_radius')
        self.grid_size = config.getfloat('sim', 'grid_size')
        self.grid_count = (self.square_width // self.grid_size) ** 2
        logging.info('grid_size:{}, Square width: {}, circle width: {}'.format(self.grid_size, self.square_width, self.circle_radius))


    def generate_random_obstacle_position(self):
        self.obstacles = []
        #obstacle_num = random.randint(10, 30)
        obstacle_num = 5
        for i in range(obstacle_num):
            self.obstacles.append(self.generate_obstacle(i))

    def generate_obstacle(self, i):
        obstacle = Obstacle()
        radius = round(random.choice(np.linspace(self.grid_size*2, self.grid_size * 5, num=6)), 2)
        x_position = round(random.choice(np.linspace(-self.square_width//2, self.square_width//2, num=int(self.square_width))), 2)
        y_position = round(random.choice(np.linspace(-self.circle_radius + 2, self.square_width//2, num=int(self.square_width))), 2)
        obstacle.set(x_position, y_position, radius)
        #logging.info('obstacle num:{}, x_position:{}, y_position:{}, radius:{}'.format(i, x_position, y_position, radius))
        return obstacle

    def set_robot(self, robot):
        self.robot = robot

    def reset(self, phase, test_case=None):
        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']

        self.global_time = 0
        self.robot.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi / 2)

        self.generate_random_obstacle_position()

        self.robot.time_step = self.time_step
        self.robot.policy.time_step = self.time_step
        self.states = list()

        if hasattr(self.robot.policy, 'action_values'):
            self.action_values = list()

        ob = [obstacle.get_state() for obstacle in self.obstacles]
        return ob

    def step(self, action, update=True):
        dmin = float('inf')
        collision = False
        for i, obstacle in enumerate(self.obstacles):
            px = obstacle.px - self.robot.px
            py = obstacle.py - self.robot.py

            vx = -action.v * np.cos(action.r + self.robot.theta)
            vy = -action.v * np.sin(action.r + self.robot.theta)

            ex = px + vx * self.time_step
            ey = py + vy * self.time_step

            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - obstacle.radius - self.robot.radius
            if closest_dist < 0:
                collision = True
                # logging.debug("Collision: distance between robot and p{} is {:.2E}".format(i, closest_dist))
                break
            elif closest_dist < dmin:
                dmin = closest_dist

        end_position = np.array(self.robot.compute_position(action, self.time_step))
        reaching_goal = norm(end_position - np.array(self.robot.get_goal_position())) < self.robot.radius + self.goal_radius

        if self.global_time >= self.time_limit - 1:
            reward = 0
            done = True
            info = Timeout()
        elif collision:
            reward = self.collision_penalty
            done = True
            info = Collision()
        elif reaching_goal:
            reward = self.success_reward
            done = True
            info = ReachGoal()
        elif dmin < self.discomfort_dist:
            # only penalize agent for getting too close if it's visible
            # adjust the reward based on FPS
            reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            info = Danger(dmin)
        else:
            reward = 0
            done = False
            info = Nothing()
        self.states.append(
            [self.robot.get_full_state(), [obstacle.get_state() for obstacle in self.obstacles]])
        if hasattr(self.robot.policy, 'action_values'):
            self.action_values.append(self.robot.policy.action_values)

            # update all agents
        self.robot.step(action)
        self.global_time += self.time_step
        ob = [obstacle.get_state() for obstacle in self.obstacles]

        return ob, reward, done, info


    def render(self, mode='human', output_file=None):
        from matplotlib import animation
        import matplotlib.pyplot as plt
        #plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

        x_offset = 0.11
        y_offset = 0.11
        cmap = plt.cm.get_cmap('hsv', 10)
        robot_color = 'yellow'
        goal_color = 'red'
        arrow_color = 'red'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)


        if mode == 'video':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            # add robot and its goal
            robot_positions = [state[0].position for state in self.states]
            goal = mlines.Line2D([0], [4], color=goal_color, marker='*', linestyle='None', markersize=15, label='Goal')
            robot = plt.Circle(robot_positions[0], self.robot.radius, fill=True, color=robot_color)
            ax.add_artist(robot)
            ax.add_artist(goal)
            plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=16)

            # add humans and their numbers
            obstacle_positions = [[state[1][j].position for j in range(len(self.obstacles))] for state in self.states]
            obstacles = [plt.Circle(obstacle_positions[0][i], self.obstacles[i].radius, fill=False)
                      for i in range(len(self.obstacles))]
            obstacle_numbers = [plt.text(obstacles[i].center[0] - x_offset, obstacles[i].center[1] - y_offset, str(i),
                                      color='black', fontsize=12) for i in range(len(self.obstacles))]
            for i, obstacle in enumerate(obstacles):
                ax.add_artist(obstacle)
                ax.add_artist(obstacle_numbers[i])

            # add time annotation
            time = plt.text(-1, 5, 'Time: {}'.format(0), fontsize=16)
            ax.add_artist(time)

            # compute attention scores

            # compute orientation in each step and use arrow to show the direction
            radius = self.robot.radius
            global_step = 0

            def update(frame_num):
                nonlocal global_step
                global_step = frame_num
                robot.center = robot_positions[frame_num]
                for i, obstacle in enumerate(obstacles):
                    obstacle.center = obstacle_positions[frame_num][i]
                    obstacle_numbers[i].set_position((obstacle.center[0] - x_offset, obstacle.center[1] - y_offset))

                time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))

            def plot_value_heatmap():
                assert self.robot.kinematics == 'holonomic'
                for agent in [self.states[global_step][0]] + self.states[global_step][1]:
                    print(('{:.4f}, ' * 6 + '{:.4f}').format(agent.px, agent.py, agent.gx, agent.gy,
                                                             agent.vx, agent.vy, agent.theta))
                # when any key is pressed draw the action value plot
                fig, axis = plt.subplots()
                speeds = [0] + self.robot.policy.speeds
                rotations = self.robot.policy.rotations + [np.pi * 2]
                r, th = np.meshgrid(speeds, rotations)
                z = np.array(self.action_values[global_step % len(self.states)][1:])
                z = (z - np.min(z)) / (np.max(z) - np.min(z))
                z = np.reshape(z, (16, 5))
                polar = plt.subplot(projection="polar")
                polar.tick_params(labelsize=16)
                mesh = plt.pcolormesh(th, r, z, vmin=0, vmax=1)
                plt.plot(rotations, r, color='k', ls='none')
                plt.grid()
                cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8])
                cbar = plt.colorbar(mesh, cax=cbaxes)
                cbar.ax.tick_params(labelsize=16)
                plt.show()

            def on_click(event):
                anim.running ^= True
                if anim.running:
                    anim.event_source.stop()
                    if hasattr(self.robot.policy, 'action_values'):
                        plot_value_heatmap()
                else:
                    anim.event_source.start()

            fig.canvas.mpl_connect('key_press_event', on_click)
            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 1000)
            anim.running = True

            if output_file is not None:
                ffmpeg_writer = animation.writers['pillow']
                #writer = animation.FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
                writer = ffmpeg_writer(fps=8, metadata=dict(artist='Me'), bitrate=1800)
                anim.save(output_file, writer=writer)
            else:
                plt.show()
        else:
            raise NotImplementedError