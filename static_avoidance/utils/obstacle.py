from utils.state import ObstacleState

class Obstacle():
    def __init__(self):
        """
        Base class for robot and human. Have the physical attributes of an agent.

        """
        self.px = None
        self.py = None
        self.radius = None

    def set(self, px, py, radius=None):
        self.px = px
        self.py = py
        if radius is not None:
            self.radius = radius

    def get_state(self):
        return ObstacleState(self.px, self.py, self.radius)
