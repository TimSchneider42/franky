from ._franky import _RobotInternal

from .desk import DeskWebSession


class Robot(_RobotInternal):
    """A class representing a Franka robot, and the main entry point of franky.

    This class allows executing motions on the robot, querying its state, and setting
    control parameters.
    """

    def create_web_session(self, username: str, password: str) -> DeskWebSession:
        """Create a web session to the Desk interface of this robot.

        Note that this method returns a :class:`DeskWebSession`, which is only compatible with
        the legacy Franka Desk API (Panda and FR3 on pre-v1 firmware). For FR3 on System 5+
        firmware, use :class:`franky.Desk` directly.

        Args:
            username: Username to log into Franka Desk.
            password: Password to log into Franka Desk.

        Returns:
            The Desk web session. Note that the session has to be opened before use, e.g. by
            using it as a context manager.
        """
        return DeskWebSession(self.fci_hostname, username, password)
