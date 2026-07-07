from ._franky import _RobotInternal

from .desk import DeskWebSession


class Robot(_RobotInternal):
    def create_web_session(self, username: str, password: str):
        return DeskWebSession(self, username, password)
