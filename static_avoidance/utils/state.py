class FullState(object):
    def __init__(self, px, py, vx, vy, radius, gx, gy, v_pref, theta):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.gx = gx
        self.gy = gy
        self.v_pref = v_pref
        self.theta = theta

        self.position = (self.px, self.py)
        self.goal_position = (self.gx, self.gy)
        self.velocity = (self.vx, self.vy)

    def __add__(self, other):
        return other + (self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy,
                                          self.v_pref, self.theta]])


class SmallState(object):
    def __init__(self, vx, vy, radius):
        self.vx = vx
        self.vy = vy
        self.radius = radius

    def __str__(self):
        return ' '.join([str(x) for x in [self.vx, self.vy, self.radius]])

class ObservableState(object):
    def __init__(self, px, py, vx, vy, radius):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius

        self.position = (self.px, self.py)
        self.velocity = (self.vx, self.vy)

    def __add__(self, other):
        return other + (self.px, self.py, self.vx, self.vy, self.radius)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius]])

class ObstacleState(object):
    def __init__(self, px, py, radius):
        self.px = px
        self.py = py
        self.radius = radius
        self.position = (self.px, self.py)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.radius]])

class JointState(object):
    def __init__(self, self_state, obstacle_states):
        assert isinstance(self_state, FullState)
        for obstacle_state in obstacle_states:
            assert isinstance(obstacle_state, ObstacleState)

        self.self_state = self_state
        self.obstacle_states = obstacle_states


