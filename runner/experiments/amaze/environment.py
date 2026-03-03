from amaze.simu import Maze, Robot, Simulation
from amaze.simu.types import InputType, OutputType
from amaze.misc.resources import qimage_to_numpy
from amaze.simu.pos import Vec
from amaze.simu.simulation import Simulation
from amaze.visu.widgets.maze import MazeWidget

from gymnasium import spaces, Env
from gymnasium import Space
from gymnasium.spaces import Discrete
from typing import Optional, List
import numpy as np
import os

from PyQt5.QtCore import QLibraryInfo
from PyQt5.QtGui import QImage, QPainter
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets


QT_PLATFORM_PLUGIN_KEY = "QT_QPA_PLATFORM"
QT_PLATFORM_OFFSCREEN_PLUGIN = "offscreen"


class IOMapper:
    """Transform AMaze's inputs/outputs types to SB3 objects"""

    def __init__(self, observation_space: Space, action_space: Space):
        self.o_space = observation_space
        if len(self.o_space.shape) == 1:
            self.map_observation = lambda obs: obs
        else:
            self.map_observation = (
                lambda obs: (obs * 255).astype(np.uint8).reshape(self.o_space.shape)
            )

        self.a_space = action_space
        if isinstance(self.a_space, Discrete):
            self.action_mapping = Simulation.discrete_actions()
            self.map_action = lambda a: Vec(*self.action_mapping[a])
        else:
            self.map_action = lambda a: Vec(*a)



class CV2QTGuard:
    """Acts as a guard allowing both PyQt5 and opencv-python to use the
    xcb.qpa plugin without confusion.

    Temporarily restores environmental variable "QT_QPA_PLATFORM_PLUGIN_PATH"
    to the value used by qt, taken from
    QLibraryInfo.location(QLibraryInfo.PluginsPath)
    """

    QPA_PATH_NAME = "QT_QPA_PLATFORM_PLUGIN_PATH"
    QPA_PLATFORM_NAME = QT_PLATFORM_PLUGIN_KEY

    def __init__(self, platform=True, path=True):
        self._qta_platform, self._qta_path = platform, path
        self.qta_platform, self.qta_path = None, None

    @staticmethod
    def _save_and_replace(key, override):
        value = os.environ.get(key, None)
        os.environ[key] = override
        return value

    def __enter__(self):
        if self._qta_platform:
            self.qta_platform = self._save_and_replace(
                self.QPA_PLATFORM_NAME, QT_PLATFORM_OFFSCREEN_PLUGIN
            )
        if self._qta_path:
            self.qta_path = self._save_and_replace(
                self.QPA_PATH_NAME,
                QLibraryInfo.location(QLibraryInfo.PluginsPath),
            )

    @staticmethod
    def _restore_or_clean(key, saved_value):
        if isinstance(saved_value, str):
            os.environ[key] = saved_value
        else:
            os.environ.pop(key)

    def __exit__(self, *_):
        if self._qta_platform:
            self._restore_or_clean(self.QPA_PLATFORM_NAME, self.qta_platform)
        if self._qta_path:
            self._restore_or_clean(self.QPA_PATH_NAME, self.qta_path)
        return False



class EnvironmentWrapper(Env):
    """AMaze wrapper for SHARPIE, based on https://amaze.readthedocs.io/en/latest/_modules/amaze/extensions/sb3/maze_env.html"""

    metadata = dict(
        render_modes=["human", "rgb_array"], render_fps=30, min_resolution=256
    )

    def __init__(
        self,
        maze: Maze.BuildData,
        robot: Robot.BuildData,
        log_trajectory: bool = False,
    ):
        """Built with maze data and robot data

        :param ~amaze.simu.maze.Maze.BuildData maze: maze data
        :param ~amaze.simu.robot.Robot.BuildData robot: agent data
        """
        super().__init__()
        self.render_mode = "rgb_array"

        self.name = maze.to_string()

        self._simulation = Simulation(
            Maze.generate(maze), robot, save_trajectory=log_trajectory
        )
        _pretty_rewards = ", ".join(
            f"{k}: {v:.2g}" for k, v in self._simulation.rewards.__dict__.items()
        )

        self.observation_type = robot.inputs
        if robot.inputs is InputType.DISCRETE:
            self.observation_space = spaces.Box(
                low=-1, high=1, shape=(8,), dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(1, robot.vision, robot.vision),
                dtype=np.uint8,
            )

        self.action_type = robot.outputs
        self.action_space = {
            OutputType.DISCRETE: spaces.Discrete(4),
            OutputType.CONTINUOUS: spaces.Box(
                low=-1, high=1, shape=(2,), dtype=np.float32
            ),
        }[robot.outputs]

        self.mapper = IOMapper(
            observation_space=self.observation_space,
            action_space=self.action_space,
        )

        self.widget, self.app = None, None

        self.prev_trajectory = None

        self.last_infos = None
        self.length = len(self._simulation.maze.solution)

        self.resets = 0

    def reset(self, seed=None, options=None, full_reset=False):
        """Stub"""
        self.last_infos = self.infos()
        if self._simulation.trajectory is not None:
            self.prev_trajectory = self._simulation.trajectory.copy(True)

        super().reset(seed=seed)
        self._simulation.reset()

        maze_str = self._simulation.maze.to_string()
        if full_reset:
            self.resets = 0
        else:
            self.resets += 1

        return self._observations(), self.infos()

    def step(self, action_dict):
        """
        Execute one step in the environment.

        Args:
            action_dict: Dictionary with agent id as keys and action as value

        Returns:
            observation: New observation (numpy array)
            reward: Reward for the action (float)
            terminated: Whether the episode has ended (bool)
            truncated: Whether the episode was truncated (bool)
            info: Additional information (dict)
        """
        # Convert dict action to discrete action (int) - AMaze is single-agent
        action = list(action_dict.values())[0]
        vec_action = self.mapper.map_action(action)

        reward = self._simulation.step(vec_action)
        observation = self._observations()
        terminated = self._simulation.success()
        truncated = self._simulation.failure()
        info = self._simulation.infos()

        # done = terminated or truncated
        # logger.debug(f"Step {self._simulation.timestep:03d} ({done=})"
        #              f" for {self._simulation.maze.to_string()}")

        return observation, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Stub"""
        with CV2QTGuard():  # Using Qt in CV2 context -> Protect
            s = 256

            if self.widget is None:
                self.widget = self._create_widget(show_robot=True)

            img = QImage(s, s, QImage.Format_RGB888)
            img.fill(Qt.white)

            painter = QPainter(img)
            self.widget.render_onto(painter, width=s)
            painter.end()

            return qimage_to_numpy(img)

    def _observations(self):
        return self.mapper.map_observation(self._simulation.observations)

    def infos(self):
        return self._simulation.infos()

    def _create_widget(self, show_robot=False):
        if self.widget:
            return self.widget

        app = QtWidgets.QApplication.instance()
        if app is None:
            # logger.debug("Creating qt app")
            self.app = QtWidgets.QApplication([])

        # logger.debug("Creating qt widget")

        self.widget = MazeWidget.from_simulation(simulation=self._simulation)
        self.widget.update_config(robot=show_robot, solution=True, dark=False)
        return self.widget



def input_mapping(inputs):
    if len(inputs) == 0:
        return 0
    
    for agent, actions in inputs.items():
        if 'ArrowUp' in actions:
            inputs[agent] = 1
        elif 'ArrowDown' in actions:
            inputs[agent] = -1
        else:
            inputs[agent] = 0

    return inputs['agent_0']

def termination_condition(terminated, truncated):
    return terminated or truncated




maze = Maze.BuildData.from_string("M16_10x10_U")
robot = Robot.BuildData.from_string("DD")
environment = EnvironmentWrapper(maze, robot)