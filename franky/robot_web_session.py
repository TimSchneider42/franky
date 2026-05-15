import base64
import hashlib
import http.client
import json
import logging
import ssl
import time
import urllib.parse
from dataclasses import dataclass
from enum import Enum
from http.client import HTTPSConnection, HTTPResponse
from typing import Dict, Optional, Any, Literal, List

import websockets.sync.client as ws_sync

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


def _encode_password(user: str, password: str) -> str:
    bs = ",".join(
        [
            str(b)
            for b in hashlib.sha256(
                (password + "#" + user + "@franka").encode("utf-8")
            ).digest()
        ]
    )
    return base64.encodebytes(bs.encode("utf-8")).decode("utf-8")


def _parse_pilot_button_payload(payload: str) -> List[PilotButtonEvent]:
    data = json.loads(payload)
    if not isinstance(data, dict):
        logger.debug(
            "Ignoring non-object event payload of type %s: %r",
            type(data).__name__,
            data,
        )
        return []
    events = []
    for key, value in data.items():
        button = PilotButton(key)
        events.append(PilotButtonEvent(button=button, pressed=bool(value)))
    return events


class _PilotMixin:
    """Shared Pilot button websocket logic for both legacy and v1 session classes."""

    _NAVIGATION_EVENTS_PATH = "/desk/api/navigation/events"
    _hostname: str  # set by subclass

    def __init__(self):
        self._pilot_button_socket = None

    @property
    def is_open(self) -> bool:
        raise NotImplementedError

    def _pilot_auth_headers(self) -> Dict[str, str]:
        raise NotImplementedError

    def poll_buttons(self, timeout: Optional[float] = 0.0) -> List[PilotButtonEvent]:
        """Poll for Pilot button events, returning all currently available.

        Args:
            timeout: Maximum seconds to wait for the first event.
                0.0 (default) returns immediately with any buffered events.
                None blocks until at least one event is available.
                After the first event arrives, any additional buffered
                events are drained without waiting.

        Returns:
            List of button events, or an empty list if none arrived
            within the timeout.
        """
        if not self.is_open:
            raise RuntimeError(
                "Session is not open. Call open() or use a context manager first."
            )

        try:
            websocket = self._get_pilot_button_socket()
            message = websocket.recv(timeout=timeout)
        except TimeoutError:
            return []
        except Exception:
            self._close_pilot_button_socket()
            raise

        events = list(_parse_pilot_button_payload(message))
        while True:
            try:
                message = websocket.recv(timeout=0.0)
            except TimeoutError:
                return events
            except Exception:
                self._close_pilot_button_socket()
                raise
            events.extend(_parse_pilot_button_payload(message))

    def _get_pilot_button_socket(self):
        if self._pilot_button_socket is None:
            uri = f"wss://{self._hostname}{self._NAVIGATION_EVENTS_PATH}"
            ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE
            self._pilot_button_socket = ws_sync.connect(
                uri,
                ssl=ssl_ctx,
                additional_headers=self._pilot_auth_headers(),
                open_timeout=2.0,
            )
        return self._pilot_button_socket

    def _close_pilot_button_socket(self) -> None:
        if self._pilot_button_socket is None:
            return
        try:
            self._pilot_button_socket.close()
        finally:
            self._pilot_button_socket = None


class RobotWebSession(_PilotMixin):
    """Robot web session for the legacy Franka Desk API.

    Compatible with Panda and FR3 on pre-v1 firmware. For FR3 on current
    firmware, use :class:`Desk` instead.
    """

    def __init__(self, hostname: str, username: str, password: str):
        super().__init__()
        self._hostname = hostname
        self.__username = username
        self.__password = password

        self.__client = None
        self.__token = None
        self.__control_token = None
        self.__control_token_id = None

    def _pilot_auth_headers(self) -> Dict[str, str]:
        return {"authorization": self.__token}

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
            self._hostname, timeout=timeout, context=ssl._create_unverified_context()
        )
        self.__client.connect()
        payload = json.dumps(
            {
                "login": self.__username,
                "password": _encode_password(self.__username, self.__password),
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
        self._close_pilot_button_socket()
        if self.__control_token is not None:
            self.release_control()
        self.__token = None
        self.__client.close()
        self.__client = None

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

    @property
    def client(self) -> HTTPSConnection:
        return self.__client

    @property
    def token(self) -> str:
        return self.__token

    @property
    def is_open(self) -> bool:
        return self.__token is not None


class Desk(_PilotMixin):
    """Robot web session for the current Franka Desk API (v1).

    Uses HTTP Basic authentication on every request. Compatible with FR3
    on System 5+ firmware. For older firmware, use :class:`RobotWebSession`.
    """

    def __init__(self, hostname: str, username: str, password: str):
        super().__init__()
        self._hostname = hostname
        self.__username = username
        self.__basic = base64.b64encode(f"{username}:{password}".encode()).decode()
        self.__client: Optional[HTTPSConnection] = None
        self.__control_token: Optional[str] = None
        self.__control_token_id: Optional[int] = None

    def _pilot_auth_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Basic {self.__basic}"}

    def _send_request(
        self,
        method: Literal["GET", "POST", "DELETE"],
        path: str,
        body: Optional[Any] = None,
        control_token: bool = False,
    ) -> bytes:
        headers: Dict[str, str] = {"Authorization": f"Basic {self.__basic}"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if control_token:
            if self.__control_token is None:
                raise RuntimeError(
                    "Client does not have control. Call take_control() first."
                )
            headers["X-Control-Token"] = self.__control_token
        encoded_body = json.dumps(body).encode() if body is not None else None
        self.__client.request(method, path, headers=headers, body=encoded_body)
        res = self.__client.getresponse()
        data = res.read()
        if res.getcode() not in (200, 204):
            raise FrankaAPIError(
                path, res.getcode(), res.reason, dict(res.headers), data.decode("utf-8")
            )
        return data

    def _request(
        self,
        method: Literal["GET", "POST", "DELETE"],
        path: str,
        body: Optional[Any] = None,
        control_token: bool = False,
    ) -> bytes:
        last_error = None
        for _ in range(3):
            try:
                return self._send_request(
                    method, path, body=body, control_token=control_token
                )
            except http.client.RemoteDisconnected as ex:
                last_error = ex
        raise last_error

    def open(self, timeout: float = 30.0):
        if self.is_open:
            raise RuntimeError("Session is already open.")
        self.__client = HTTPSConnection(
            self._hostname, timeout=timeout, context=ssl._create_unverified_context()
        )
        self.__client.connect()
        self._request("GET", "/api/system")
        return self

    def close(self):
        if not self.is_open:
            raise RuntimeError("Session is not open.")
        self._close_pilot_button_socket()
        if self.__control_token is not None:
            self.release_control()
        self.__client.close()
        self.__client = None

    def __enter__(self):
        return self.open()

    def __exit__(self, type, value, traceback):
        self.close()

    def take_control(self, wait_timeout: float = 30.0, force: bool = False):
        """Request the control token.

        Args:
            wait_timeout: Seconds to wait if another client currently holds
                the token. The server handles the wait; has no effect if we
                already hold the token.
            force: Ignored on v1 API — the server queues the request until
                the current holder releases.
        """
        if self.has_control():
            return
        res = json.loads(
            self._request(
                "POST",
                "/api/system/control-token:take",
                body={"owner": self.__username, "timeout": int(wait_timeout)},
            )
        )
        self.__control_token = res["token"]
        self.__control_token_id = res["tokenId"]

    def release_control(self):
        if self.__control_token is None:
            return
        self._request("POST", "/api/system/control-token:release", control_token=True)
        self.__control_token = None
        self.__control_token_id = None

    def has_control(self) -> bool:
        if self.__control_token_id is None:
            return False
        data = json.loads(self._request("GET", "/api/system/control-token"))
        return data.get("tokenId") == self.__control_token_id

    def unlock_brakes(self):
        self._request("POST", "/api/arm/joints:unlock", control_token=True)

    def lock_brakes(self):
        self._request("POST", "/api/arm/joints:lock")

    def set_mode_programming(self):
        self._request(
            "POST",
            "/api/system/operating-mode:change",
            body={"desiredOperatingMode": "Programming"},
            control_token=True,
        )

    def set_mode_execution(self):
        self._request(
            "POST",
            "/api/system/operating-mode:change",
            body={"desiredOperatingMode": "Execution"},
            control_token=True,
        )

    def enable_fci(self):
        self._request("POST", "/api/fci:activate", control_token=True)

    def disable_fci(self):
        self._request("POST", "/api/fci:deactivate", control_token=True)

    def get_fci_status(self) -> str:
        return json.loads(self._request("GET", "/api/fci"))["status"]

    def get_system_status(self) -> dict:
        return json.loads(self._request("GET", "/api/system"))

    def execute_self_test(self):
        self._request("POST", "/api/safety/self-tests:execute", control_token=True)
        while (
            json.loads(self._request("GET", "/api/safety/self-tests"))["status"]
            == "Running"
        ):
            time.sleep(0.5)

    def get_recovery_status(self) -> Optional[dict]:
        """Return the active recovery descriptor, or None if no recovery is needed."""
        data = json.loads(self._request("GET", "/api/safety/recovery"))
        return data.get("recovery")

    def recover(self) -> None:
        """Attempt to clear an active recoverable safety error.

        Handles errors that can be confirmed programmatically
        (LifetimeStatusExceeded, SelfTestsElapsed, GenericJointError,
        SafetyError, SafeInputUnacknowledged, SafetyRuleViolation).
        Raises :class:`RobotWebSessionError` for errors that require
        physical joint movement (JointPositionError, JointLimitViolation).
        Does nothing if no recovery is needed.
        """
        recovery = self.get_recovery_status()
        if recovery is None:
            return
        recovery_type = recovery["type"]
        if recovery_type in ("JointPositionError", "JointLimitViolation"):
            raise RobotWebSessionError(
                f"Recovery type '{recovery_type}' requires physical joint movement. "
                "Move the affected joints to their reference positions and confirm "
                "via the Desk interface. Consult the product manual for details."
            )
        body: Dict[str, Any] = {"type": recovery_type}
        if recovery_type == "SafetyError":
            body["safetyErrors"] = recovery.get("safetyErrors", [])
        elif recovery_type == "SafeInputUnacknowledged":
            body["safeInputs"] = recovery.get("safeInputs", [])
        self._request("POST", "/api/safety/recovery:confirm", body=body)

    def reboot(self) -> None:
        """Initiate a system reboot. Closes all open connections."""
        self._request("POST", "/api/system:reboot")

    def shutdown(self) -> None:
        """Initiate a system shutdown."""
        self._request("POST", "/api/system:shutdown")

    @property
    def is_open(self) -> bool:
        return self.__client is not None
