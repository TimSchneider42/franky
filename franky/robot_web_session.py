import base64
import hashlib
import http.client
import json
import logging
import ssl
import threading
import time
import urllib.parse
from dataclasses import dataclass
from enum import Enum
from http.client import HTTPSConnection, HTTPResponse
from typing import Dict, Optional, Any, Literal, Callable

try:
    import websockets.sync.client as ws_sync
except ImportError:
    ws_sync = None

logger = logging.getLogger(__name__)


class RobotWebSessionError(Exception):
    pass


class FrankaAPIError(RobotWebSessionError):
    def __init__(
        self,
        target: str,
        http_code: int,
        http_reason: str,
        headers: Dict[str, str],
        message: str,
    ):
        super().__init__(
            f"Franka API returned error {http_code} ({http_reason}) when accessing end-point {target}: {message}"
        )
        self.target = target
        self.http_code = http_code
        self.headers = headers
        self.message = message


class TakeControlTimeoutError(RobotWebSessionError):
    pass


class PilotButton(Enum):
    """Buttons on the Franka Desk Pilot interface."""

    CIRCLE = "circle"
    CROSS = "cross"
    CHECK = "check"
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True)
class PilotButtonEvent:
    """A single Pilot button state change."""

    button: PilotButton
    pressed: bool

    def __repr__(self) -> str:
        action = "pressed" if self.pressed else "released"
        return f"PilotButtonEvent({self.button.value} {action})"


class RobotWebSession:
    _NAVIGATION_EVENTS_PATH = "/desk/api/navigation/events"

    def __init__(self, hostname: str, username: str, password: str):
        self.__hostname = hostname
        self.__username = username
        self.__password = password

        self.__client = None
        self.__token = None
        self.__control_token = None
        self.__control_token_id = None
        self.__listener_thread: Optional[threading.Thread] = None
        self.__stop_event = threading.Event()

    @classmethod
    def from_session(cls, session: "RobotWebSession") -> "RobotWebSession":
        """Return an already-open session for API compatibility with Desk."""
        if not session.is_open:
            raise RuntimeError("The provided RobotWebSession is not open.")
        return session

    @staticmethod
    def __encode_password(user: str, password: str) -> str:
        bs = ",".join(
            [
                str(b)
                for b in hashlib.sha256(
                    (password + "#" + user + "@franka").encode("utf-8")
                ).digest()
            ]
        )
        return base64.encodebytes(bs.encode("utf-8")).decode("utf-8")

    def _send_api_request(
        self,
        target: str,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Any] = None,
        method: Literal["GET", "POST", "DELETE"] = "POST",
    ):
        _headers = {"Cookie": f"authorization={self.__token}"}
        if headers is not None:
            _headers.update(headers)
        self.__client.request(method, target, headers=_headers, body=body)
        res: HTTPResponse = self.__client.getresponse()
        if res.getcode() != 200:
            raise FrankaAPIError(
                target,
                res.getcode(),
                res.reason,
                dict(res.headers),
                res.read().decode("utf-8"),
            )
        return res.read()

    def send_api_request(
        self,
        target: str,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Any] = None,
        method: Literal["GET", "POST", "DELETE"] = "POST",
    ):
        last_error = None
        for i in range(3):
            try:
                return self._send_api_request(target, headers, body, method)
            except http.client.RemoteDisconnected as ex:
                last_error = ex
        raise last_error

    def send_control_api_request(
        self,
        target: str,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Any] = None,
        method: Literal["GET", "POST", "DELETE"] = "POST",
    ):
        if headers is None:
            headers = {}
        self.__check_control_token()
        _headers = {"X-Control-Token": self.__control_token}
        _headers.update(headers)
        return self.send_api_request(target, headers=_headers, method=method, body=body)

    def open(self, timeout: float = 30.0):
        if self.is_open:
            raise RuntimeError("Session is already open.")
        self.__client = HTTPSConnection(
            self.__hostname, timeout=timeout, context=ssl._create_unverified_context()
        )
        self.__client.connect()
        payload = json.dumps(
            {
                "login": self.__username,
                "password": self.__encode_password(self.__username, self.__password),
            }
        )
        self.__token = self.send_api_request(
            "/admin/api/login",
            headers={"content-type": "application/json"},
            body=payload,
        ).decode("utf-8")
        return self

    def close(self):
        if not self.is_open:
            raise RuntimeError("Session is not open.")
        self.stop_listen()
        if self.__control_token is not None:
            self.release_control()
        self.__token = None
        self.__client.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, type, value, traceback):
        self.close()

    def __check_control_token(self):
        if self.__control_token is None:
            raise RuntimeError(
                "Client does not have control. Call take_control() first."
            )

    def take_control(self, wait_timeout: float = 30.0, force: bool = False):
        if not self.has_control():
            res = self.send_api_request(
                f"/admin/api/control-token/request{'?force' if force else ''}",
                headers={"content-type": "application/json"},
                body=json.dumps({"requestedBy": self.__username}),
            )
            if force:
                print(
                    "Forcibly taking control: "
                    f"Please physically take control by pressing the top button on the FR3 within {wait_timeout}s!"
                )
            response_dict = json.loads(res)
            self.__control_token = response_dict["token"]
            self.__control_token_id = response_dict["id"]
            # One should probably use websockets here but that would introduce another dependency
            start = time.time()
            has_control = self.has_control()
            while time.time() - start < wait_timeout and not has_control:
                time.sleep(max(0.0, min(1.0, wait_timeout - (time.time() - start))))
                has_control = self.has_control()
            if not has_control:
                raise TakeControlTimeoutError(
                    f"Timed out waiting for control to be granted after {wait_timeout}s."
                )

    def release_control(self):
        if self.__control_token is not None:
            self.send_control_api_request(
                "/admin/api/control-token",
                headers={"content-type": "application/json"},
                method="DELETE",
                body=json.dumps({"token": self.__control_token}),
            )
            self.__control_token = None
            self.__control_token_id = None

    def enable_fci(self):
        self.send_control_api_request(
            "/desk/api/system/fci",
            headers={"content-type": "application/x-www-form-urlencoded"},
            body=f"token={urllib.parse.quote(base64.b64encode(self.__control_token.encode('ascii')))}",
        )

    def has_control(self):
        if self.__control_token_id is not None:
            status = self.get_system_status()
            active_token = status["controlToken"]["activeToken"]
            return (
                active_token is not None
                and active_token["id"] == self.__control_token_id
            )
        return False

    def start_task(self, task: str):
        self.send_api_request(
            "/desk/api/execution",
            headers={"content-type": "application/x-www-form-urlencoded"},
            body=f"id={task}",
        )

    def unlock_brakes(self):
        self.send_control_api_request(
            "/desk/api/joints/unlock",
            headers={"content-type": "application/x-www-form-urlencoded"},
        )

    def lock_brakes(self):
        self.send_control_api_request(
            "/desk/api/joints/lock",
            headers={"content-type": "application/x-www-form-urlencoded"},
        )

    def set_mode_programming(self):
        self.send_control_api_request(
            "/desk/api/operating-mode/programming",
            headers={"content-type": "application/x-www-form-urlencoded"},
        )

    def set_mode_execution(self):
        self.send_control_api_request(
            "/desk/api/operating-mode/execution",
            headers={"content-type": "application/x-www-form-urlencoded"},
        )

    def get_system_status(self):
        return json.loads(
            self.send_api_request("/admin/api/system-status", method="GET").decode(
                "utf-8"
            )
        )

    def execute_self_test(self):
        if self.get_system_status()["safety"]["recoverableErrors"]["td2Timeout"]:
            self.send_control_api_request(
                "/admin/api/safety/recoverable-safety-errors/acknowledge?error_id=TD2Timeout"
            )
        response = json.loads(
            self.send_control_api_request(
                "/admin/api/safety/td2-tests/execute",
                headers={"content-type": "application/json"},
            ).decode("utf-8")
        )
        assert response["code"] == "SuccessResponse"
        time.sleep(0.5)
        while (
            self.get_system_status()["safety"]["safetyControllerStatus"] == "SelfTest"
        ):
            time.sleep(0.5)

    def listen(self, callback: Callable[[PilotButtonEvent], None]) -> None:
        """Start listening for Pilot button events in a background thread."""
        if ws_sync is None:
            raise RuntimeError(
                "The 'websockets' package is required for Pilot button listening. "
                "Install it with: pip install websockets"
            )
        if self.__listener_thread is not None and self.__listener_thread.is_alive():
            raise RuntimeError(
                "Already listening. Call stop_listen() before starting a new listener."
            )
        if not self.is_open:
            raise RuntimeError(
                "Session is not open. Call open() or use a context manager first."
            )

        self.__stop_event.clear()
        self.__listener_thread = threading.Thread(
            target=self._listen_loop,
            args=(callback,),
            daemon=True,
            name="franky-pilot-listener",
        )
        self.__listener_thread.start()

    def stop_listen(self) -> None:
        """Stop the background Pilot button listener, if running."""
        if self.__listener_thread is not None and self.__listener_thread.is_alive():
            self.__stop_event.set()
            self.__listener_thread.join(timeout=5.0)
            if self.__listener_thread.is_alive():
                logger.warning("Pilot listener thread did not shut down cleanly.")
        self.__listener_thread = None

    @property
    def is_listening(self) -> bool:
        """Whether the Pilot button listener is currently active."""
        return self.__listener_thread is not None and self.__listener_thread.is_alive()

    def _listen_loop(self, callback: Callable[[PilotButtonEvent], None]) -> None:
        """Websocket receive loop executed in the listener thread."""
        uri = f"wss://{self.__hostname}{self._NAVIGATION_EVENTS_PATH}"
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        try:
            with ws_sync.connect(
                uri,
                ssl=ssl_context,
                additional_headers={"authorization": self.__token},
                open_timeout=2.0,
            ) as websocket:
                while not self.__stop_event.is_set():
                    try:
                        message = websocket.recv(timeout=1.0)
                    except TimeoutError:
                        continue
                    self._emit_payload(message, callback)
        except Exception:
            if not self.__stop_event.is_set():
                logger.exception("Pilot button listener encountered an error.")

    def _emit_payload(
        self, payload: str, callback: Callable[[PilotButtonEvent], None]
    ) -> None:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            logger.warning("Received non-JSON event payload: %s", payload)
            return

        if not isinstance(data, dict):
            logger.debug(
                "Ignoring non-object event payload of type %s: %r",
                type(data).__name__,
                data,
            )
            return

        for key, value in data.items():
            try:
                button = PilotButton(key)
            except ValueError:
                logger.debug("Unknown button key in event: %s", key)
                continue
            event = PilotButtonEvent(button=button, pressed=bool(value))
            try:
                callback(event)
            except Exception:
                logger.exception("Exception in Pilot button callback for event %s", event)

    @property
    def client(self) -> HTTPSConnection:
        return self.__client

    @property
    def token(self) -> str:
        return self.__token

    @property
    def is_open(self) -> bool:
        return self.__token is not None


Desk = RobotWebSession
