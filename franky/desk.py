from __future__ import annotations

import base64
import hashlib
import http.client
import json
import logging
import ssl
import time
import urllib.parse
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from http.client import HTTPSConnection, HTTPResponse
from typing import Optional, Any, Literal

import websockets.sync.client as ws_sync

logger = logging.getLogger(__name__)


class DeskError(Exception):
    """Base class for errors raised by the Desk classes."""


class FrankaAPIError(DeskError):
    """Raised when the Franka web API returns an error response."""

    def __init__(
        self,
        path: str,
        http_code: int,
        http_reason: str,
        headers: dict[str, str],
        message: str,
    ):
        super().__init__(
            f"Franka API returned error {http_code} ({http_reason}) when accessing end-point {path}: {message}"
        )
        self.path = path
        self.http_code = http_code
        self.headers = headers
        self.message = message


class TakeControlTimeoutError(DeskError):
    """Raised when control over the robot could not be obtained within the timeout."""


class PilotButton(Enum):
    """Buttons on the Franka Desk Pilot interface."""

    CIRCLE = "circle"
    CROSS = "cross"
    CHECK = "check"
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


class OperatingMode(Enum):
    """Operating mode of the robot as reported by Franka Desk."""

    EXECUTION = "Execution"
    PROGRAMMING = "Programming"


class BrakeState(Enum):
    """State of the robot's joint brakes."""

    LOCKED = "Locked"
    UNLOCKED = "Unlocked"


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


class BaseDesk(ABC):
    """Base class for sessions with the Franka Desk web interface.

    The Desk classes allow controlling robot functions that are otherwise only available through
    the Franka Desk web interface, such as unlocking the brakes, enabling the FCI, or reacting to
    Pilot button presses.

    Args:
        hostname: Hostname or IP address of the robot.
        username: Username to log into Franka Desk.
        password: Password to log into Franka Desk.
    """

    def __init__(self, hostname: str, username: str, password: str):
        super().__init__()
        self.__hostname = hostname
        self.__username = username
        self.__password = password
        self.__pilot_button_socket = None
        self.__client = None
        self.__control_token = None
        self.__control_token_id = None

    def open(self, timeout: float = 30.0):
        """Open the connection to Franka Desk and log in.

        Instead of calling this method directly, the session can also be used as a context
        manager.

        Args:
            timeout: Connection timeout [s].
        """
        if self.is_open:
            raise RuntimeError("Session is already open.")
        self.__client = HTTPSConnection(
            self.hostname, timeout=timeout, context=ssl._create_unverified_context()
        )
        self.__client.connect()

    def close(self):
        """Release control if held, and close the connection to Franka Desk."""
        if not self.is_open:
            raise RuntimeError("Session is not open.")
        self._close_pilot_button_socket()
        if self.__control_token is not None:
            self.release_control()
        self.__client.close()
        self.__client = None

    def take_control(self, wait_timeout: float = 30.0, force: bool = False):
        """Obtain exclusive control over the robot.

        Control is required for all operations that change the state of the robot, such as
        unlocking the brakes or enabling the FCI.

        Args:
            wait_timeout: Maximum time to wait for control to be granted [s].
            force: Whether to take control even if another user currently holds it. Depending on
                the API version, forcibly taking control may require pressing a physical button
                on the robot within the timeout.

        Raises:
            TakeControlTimeoutError: If control was not granted within the timeout.
        """
        if not self.has_control:
            self.__control_token_id, self.__control_token = self._take_control(
                wait_timeout=wait_timeout, force=force
            )

    def release_control(self):
        """Release control over the robot, allowing other users to take it."""
        if self.__control_token is not None:
            self._release_control()
        self.__control_token = None
        self.__control_token_id = None

    def send_api_request(
        self,
        path: str,
        headers: Optional[dict[str, str]] = None,
        content: dict[str, Any] | None = None,
        content_encoding: Literal["json", "x-www-form-urlencoded"] = "json",
        method: Literal["GET", "POST", "DELETE"] = "POST",
        response_encoding: Optional[Literal["json", "text"]] = None,
    ) -> Any:
        """Send a request to the Franka web API and parse the response.

        Args:
            path: Path of the API endpoint, e.g. "/api/system".
            headers: Additional HTTP headers. Overrides the content-type header derived from
                content_encoding if set.
            content: Request payload, encoded according to content_encoding.
            content_encoding: How to encode the payload.
            method: HTTP method to use.
            response_encoding: How to decode the response. If None, it is derived from the
                response content type.

        Returns:
            The parsed response, or None if the response is empty.

        Raises:
            FrankaAPIError: If the API returns an error response.
        """
        return self.__parse_response(
            *self.__request(
                path,
                method=method,
                **self.__encode_content(
                    headers=headers, content=content, content_encoding=content_encoding
                ),
            ),
            response_encoding=response_encoding,
        )

    def send_control_api_request(
        self,
        path: str,
        headers: Optional[dict[str, str]] = None,
        content: dict[str, Any] | None = None,
        content_encoding: Literal["json", "x-www-form-urlencoded"] = "json",
        method: Literal["GET", "POST", "DELETE"] = "POST",
        response_encoding: Optional[Literal["json", "text"]] = None,
    ) -> Any:
        """Send a request that requires control over the robot to the Franka web API.

        Same as send_api_request, but includes the control token in the request. Requires
        take_control to have been called first.
        """
        return self.__parse_response(
            *self.__request_with_control(
                path,
                method=method,
                **self.__encode_content(
                    headers=headers, content=content, content_encoding=content_encoding
                ),
            ),
            response_encoding=response_encoding,
        )

    def send_raw_api_request(
        self,
        path: str,
        headers: Optional[dict[str, str]] = None,
        body: Optional[Any] = None,
        method: Literal["GET", "POST", "DELETE"] = "POST",
    ) -> bytes:
        """Send a request with a raw body to the Franka web API and return the raw response."""
        return self.__request(path, headers, body, method)[0]

    def send_raw_control_api_request(
        self,
        path: str,
        headers: Optional[dict[str, str]] = None,
        body: Optional[Any] = None,
        method: Literal["GET", "POST", "DELETE"] = "POST",
    ) -> bytes:
        """Send a raw request that requires control over the robot to the Franka web API.

        Same as send_raw_api_request, but includes the control token in the request. Requires
        take_control to have been called first.
        """
        return self.__request_with_control(path, headers, body, method)[0]

    def __request(
        self,
        path: str,
        headers: Optional[dict[str, str]] = None,
        body: Optional[Any] = None,
        method: Literal["GET", "POST", "DELETE"] = "POST",
    ) -> tuple[bytes, Optional[str]]:
        last_error = None
        for i in range(3):
            try:
                return self._send_raw_api_request(path, headers, body, method)
            except http.client.RemoteDisconnected as ex:
                last_error = ex
        raise last_error

    def __request_with_control(
        self,
        path: str,
        headers: Optional[dict[str, str]] = None,
        body: Optional[Any] = None,
        method: Literal["GET", "POST", "DELETE"] = "POST",
    ) -> tuple[bytes, Optional[str]]:
        if headers is None:
            headers = {}
        self.__check_control_token()
        _headers = {"X-Control-Token": self.__control_token}
        _headers.update(headers)
        return self.__request(path, headers=_headers, method=method, body=body)

    def poll_buttons(self, timeout: Optional[float] = 0.0) -> list[PilotButtonEvent]:
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

        events = list(self.__parse_pilot_button_payload(message))
        while True:
            try:
                message = websocket.recv(timeout=0.0)
            except TimeoutError:
                return events
            except Exception:
                self._close_pilot_button_socket()
                raise
            events.extend(self.__parse_pilot_button_payload(message))

    @abstractmethod
    def _get_pilot_auth_headers(self) -> dict[str, str]:
        pass

    @abstractmethod
    def _get_default_headers(self) -> dict[str, str]:
        pass

    @abstractmethod
    def _take_control(
        self, wait_timeout: float = 30.0, force: bool = False
    ) -> tuple[str, str]:
        pass

    @abstractmethod
    def _get_has_control(self):
        pass

    @abstractmethod
    def _release_control(self):
        pass

    @abstractmethod
    def _get_system_status(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def _get_operating_mode(self) -> OperatingMode:
        pass

    @abstractmethod
    def _get_brake_state(self) -> BrakeState:
        pass

    @abstractmethod
    def enable_fci(self):
        """Enable the Franka Control Interface (FCI). Requires control over the robot."""

    @abstractmethod
    def unlock_brakes(self):
        """Unlock the joint brakes. Requires control over the robot."""

    @abstractmethod
    def lock_brakes(self):
        """Lock the joint brakes. Requires control over the robot."""

    @abstractmethod
    def set_mode_execution(self):
        """Set the operating mode to Execution. Requires control over the robot."""

    @abstractmethod
    def execute_self_test(self):
        """Execute the safety self-test and wait for it to finish. Requires control over the
        robot."""

    def _send_raw_api_request(
        self,
        path: str,
        headers: Optional[dict[str, str]] = None,
        body: Optional[Any] = None,
        method: Literal["GET", "POST", "DELETE"] = "POST",
    ) -> tuple[bytes, Optional[str]]:
        _headers = self._get_default_headers()
        if headers is not None:
            _headers.update(headers)
        self.__client.request(method, path, headers=_headers, body=body)
        res: HTTPResponse = self.__client.getresponse()
        if res.status not in (200, 204):
            raise FrankaAPIError(
                path,
                res.status,
                res.reason,
                dict(res.headers),
                res.read().decode("utf-8"),
            )
        return res.read(), res.getheader("Content-Type")

    @staticmethod
    def __parse_response(
        body: bytes,
        content_type: Optional[str],
        response_encoding: Optional[Literal["json", "text"]] = None,
    ) -> Any:
        if not body:
            return None
        if response_encoding is None:
            media_type = (content_type or "").split(";", 1)[0].strip().lower()
            if media_type == "application/json" or media_type.endswith("+json"):
                response_encoding = "json"
            else:
                response_encoding = "text"
        if response_encoding == "json":
            return json.loads(body)
        return body.decode("utf-8")

    @staticmethod
    def __encode_content(
        headers: Optional[dict[str, str]] = None,
        content: dict[str, Any] | None = None,
        content_encoding: Literal["json", "x-www-form-urlencoded"] = "json",
    ):
        if headers is None:
            headers = {"content-type": f"application/{content_encoding}"}
        if content is None:
            body = None
        else:
            if content_encoding == "json":
                body = json.dumps(content)
            elif content_encoding == "x-www-form-urlencoded":
                body = urllib.parse.urlencode(content)
            else:
                raise ValueError(f"Unsupported content encoding: {content_encoding}")
        return {"body": body, "headers": headers}

    def __check_control_token(self):
        if self.__control_token is None:
            raise RuntimeError(
                "Client does not have control. Call take_control() first."
            )

    def _get_pilot_button_socket(self):
        if self.__pilot_button_socket is None:
            uri = f"wss://{self.__hostname}/desk/api/navigation/events"
            ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE
            self.__pilot_button_socket = ws_sync.connect(
                uri,
                ssl=ssl_ctx,
                additional_headers=self._get_pilot_auth_headers(),
                open_timeout=2.0,
            )
        return self.__pilot_button_socket

    def _close_pilot_button_socket(self) -> None:
        if self.__pilot_button_socket is None:
            return
        try:
            self.__pilot_button_socket.close()
        finally:
            self.__pilot_button_socket = None

    @staticmethod
    def __parse_pilot_button_payload(payload: str) -> list[PilotButtonEvent]:
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

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    @property
    def is_open(self) -> bool:
        """Whether the session is currently open."""
        return self.__client is not None

    @property
    def hostname(self) -> str:
        """Hostname or IP address of the robot."""
        return self.__hostname

    @property
    def username(self) -> str:
        """Username used to log into Franka Desk."""
        return self.__username

    @property
    def password(self) -> str:
        """Password used to log into Franka Desk."""
        return self.__password

    @property
    def control_token(self) -> str | None:
        """The current control token, or None if this session does not hold control."""
        return self.__control_token

    @property
    def control_token_id(self) -> str | None:
        """The ID of the current control token, or None if this session does not hold control."""
        return self.__control_token_id

    @property
    def client(self) -> HTTPSConnection:
        """The underlying HTTPS connection to the robot."""
        return self.__client

    @property
    def has_control(self):
        """Whether this session currently holds control over the robot."""
        if self.__control_token_id is not None:
            return self._get_has_control()
        return False

    @property
    def system_status(self) -> dict[str, Any]:
        """The system status as reported by the Franka web API."""
        return self._get_system_status()

    @property
    def operating_mode(self) -> OperatingMode:
        """The current operating mode of the robot."""
        return self._get_operating_mode()

    @property
    def brake_state(self) -> BrakeState:
        """The current state of the joint brakes."""
        return self._get_brake_state()


class NoTokenIdType:
    pass


NO_TOKEN_ID = NoTokenIdType()


class DeskWebSession(BaseDesk):
    """Desk web session for the legacy Franka Desk API.

    Compatible with Panda and FR3 on pre-v1 firmware. For FR3 on current
    firmware, use :class:`Desk` instead.
    """

    def __init__(self, hostname: str, username: str, password: str):
        super().__init__(hostname, username, password)
        self.__token = None

    def _get_pilot_auth_headers(self) -> dict[str, str]:
        return {"authorization": self.__token}

    def _get_default_headers(self) -> dict[str, str]:
        headers = {}
        if self.__token is not None:
            headers["Cookie"] = f"authorization={self.__token}"
        return headers

    def open(self, timeout: float = 30.0):
        super().open(timeout=timeout)
        try:
            self.__token = self.send_api_request(
                "/admin/api/login",
                content={
                    "login": self.username,
                    "password": _encode_password(self.username, self.password),
                },
                response_encoding="text",
            )
        except:
            self.close()
            raise

    def close(self):
        super().close()
        self.__token = None

    def _take_control(self, wait_timeout: float = 30.0, force: bool = False):
        response_dict = self.send_api_request(
            f"/admin/api/control-token/request{'?force' if force else ''}",
            content={"requestedBy": self.username},
        )
        if force:
            print(
                "Forcibly taking control: "
                f"Please physically take control by pressing the top button on the FR3 within {wait_timeout}s!"
            )
        token_id, token = response_dict["id"], response_dict["token"]

        # Cannot use self.has_control here as the token is only stored in the
        # base class once this method returns.
        def granted() -> bool:
            active_token = self._get_system_status()["controlToken"]["activeToken"]
            return active_token is not None and active_token["id"] == token_id

        start = time.time()
        has_control = granted()
        while time.time() - start < wait_timeout and not has_control:
            time.sleep(max(0.0, min(1.0, wait_timeout - (time.time() - start))))
            has_control = granted()
        if not has_control:
            raise TakeControlTimeoutError(
                f"Timed out waiting for control to be granted after {wait_timeout}s."
            )
        return token_id, token

    def _get_has_control(self):
        status = self._get_system_status()
        active_token = status["controlToken"]["activeToken"]
        return active_token is not None and active_token["id"] == self.control_token_id

    def _release_control(self):
        self.send_control_api_request(
            "/admin/api/control-token",
            method="DELETE",
            content={"token": self.control_token},
        )

    def _get_operating_mode(self) -> OperatingMode:
        return OperatingMode(self._get_system_status()["derived"]["operatingMode"])

    def _get_brake_state(self) -> BrakeState:
        status = self._get_system_status()
        if all(b == "Unlocked" for b in status["safety"]["brakeState"]):
            return BrakeState.UNLOCKED
        elif all(b == "Locked" for b in status["safety"]["brakeState"]):
            return BrakeState.LOCKED
        else:
            raise DeskError(
                "Inconsistent brake state: some joints are locked while others are unlocked."
            )

    def enable_fci(self):
        self.send_control_api_request(
            "/desk/api/system/fci",
            content={
                "token": base64.b64encode(self.control_token.encode("ascii")).decode(
                    "ascii"
                )
            },
            content_encoding="x-www-form-urlencoded",
        )

    def start_task(self, task: str):
        """Start a task on Franka Desk.

        Args:
            task: ID of the task to start.
        """
        self.send_api_request(
            "/desk/api/execution",
            content={"id": task},
            content_encoding="x-www-form-urlencoded",
        )

    def unlock_brakes(self):
        self.send_control_api_request(
            "/desk/api/joints/unlock",
            content_encoding="x-www-form-urlencoded",
        )

    def lock_brakes(self):
        self.send_control_api_request(
            "/desk/api/joints/lock",
            content_encoding="x-www-form-urlencoded",
        )

    def set_mode_programming(self):
        """Set the operating mode to Programming. Requires control over the robot."""
        self.send_control_api_request(
            "/desk/api/operating-mode/programming",
            content_encoding="x-www-form-urlencoded",
        )

    def set_mode_execution(self):
        self.send_control_api_request(
            "/desk/api/operating-mode/execution",
            content_encoding="x-www-form-urlencoded",
        )

    def _get_system_status(self):
        return self.send_api_request("/admin/api/system-status", method="GET")

    def execute_self_test(self):
        if self._get_system_status()["safety"]["recoverableErrors"]["td2Timeout"]:
            self.send_control_api_request(
                "/admin/api/safety/recoverable-safety-errors/acknowledge?error_id=TD2Timeout"
            )
        response = self.send_control_api_request(
            "/admin/api/safety/td2-tests/execute",
        )
        assert response["code"] == "SuccessResponse"
        time.sleep(0.5)
        while (
            self._get_system_status()["safety"]["safetyControllerStatus"] == "SelfTest"
        ):
            time.sleep(0.5)

    @property
    def token(self) -> str:
        """The authentication token of this session."""
        return self.__token


class Desk(BaseDesk):
    """Robot web session for the current Franka Desk API (v1).

    Uses HTTP Basic authentication on every request. Compatible with FR3
    on System 5+ firmware. For older firmware, use :class:`DeskWebSession`.
    """

    def __init__(self, hostname: str, username: str, password: str):
        super().__init__(hostname, username, password)
        self.__warned_no_token_id = False

    def _get_pilot_auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Basic {self.__auth_string}"}

    def _get_default_headers(self) -> dict[str, str]:
        return {"Authorization": f"Basic {self.__auth_string}"}

    def open(self, timeout: float = 30.0):
        super().open(timeout=timeout)
        try:
            # Verify connectivity and credentials right away.
            self._get_system_status()
        except:
            self.close()
            raise

    def _take_control(self, wait_timeout: float = 30.0, force: bool = False):
        # force is ignored on the v1 API -- the server queues the request
        # (up to wait_timeout seconds) until the current holder releases.
        res = self.send_api_request(
            "/api/system/control-token:take",
            content={"owner": self.username, "timeout": int(wait_timeout)},
        )
        return res.get("tokenId", NO_TOKEN_ID), res["token"]

    def _release_control(self):
        self.send_control_api_request("/api/system/control-token:release")

    def _get_has_control(self) -> bool:
        data = self.send_api_request("/api/system/control-token", method="GET")
        if self.control_token_id == NO_TOKEN_ID:
            if not self.__warned_no_token_id:
                logger.warning(
                    "Your Franka API does not support checking if the current token is still valid. Falling back to "
                    "checking for just the owner. Upgrade the Franka firmware or resort to franky.DeskWebSession to "
                    "resolve this issue."
                )
                self.__warned_no_token_id = True
            return data.get("owner") == self.username
        return data.get("tokenId") == self.control_token_id

    def _get_operating_mode(self) -> OperatingMode:
        data = self.send_api_request("/api/system/operating-mode", method="GET")
        return OperatingMode(data["status"])

    def _get_brake_state(self) -> BrakeState:
        data = self.send_api_request("/api/arm/joints", method="GET")
        if all(j["brakeStatus"] == "Unlocked" for j in data):
            return BrakeState.UNLOCKED
        elif all(j["brakeStatus"] == "Locked" for j in data):
            return BrakeState.LOCKED
        else:
            raise DeskError(
                "Inconsistent brake state: some joints are locked while others are unlocked."
            )

    def enable_fci(self):
        self.send_control_api_request("/api/fci:activate")

    def disable_fci(self):
        """Disable the Franka Control Interface (FCI). Requires control over the robot."""
        self.send_control_api_request("/api/fci:deactivate")

    def get_fci_status(self) -> str:
        """Return the current status of the Franka Control Interface (FCI)."""
        return self.send_api_request("/api/fci", method="GET")["status"]

    def unlock_brakes(self):
        self.send_control_api_request("/api/arm/joints:unlock")

    def lock_brakes(self):
        self.send_control_api_request("/api/arm/joints:lock")

    def set_mode_execution(self):
        self.send_control_api_request(
            "/api/system/operating-mode:change",
            content={"desiredOperatingMode": "Execution"},
        )

    def _get_system_status(self) -> dict:
        return self.send_api_request("/api/system", method="GET")

    def execute_self_test(self):
        self.send_control_api_request("/api/safety/self-tests:execute")
        while (
            self.send_api_request("/api/safety/self-tests", method="GET")["status"]
            == "Running"
        ):
            time.sleep(0.5)

    def get_recovery_status(self) -> Optional[dict]:
        """Return the active recovery descriptor, or None if no recovery is needed."""
        data = self.send_api_request("/api/safety/recovery", method="GET")
        return data.get("recovery")

    def recover(self) -> None:
        """Attempt to clear an active recoverable safety error.

        Handles errors that can be confirmed programmatically
        (LifetimeStatusExceeded, SelfTestsElapsed, GenericJointError,
        SafetyError, SafeInputUnacknowledged, SafetyRuleViolation).
        Raises :class:`DeskError` for errors that require
        physical joint movement (JointPositionError, JointLimitViolation).
        Does nothing if no recovery is needed.
        """
        recovery = self.get_recovery_status()
        if recovery is None:
            return
        recovery_type = recovery["type"]
        if recovery_type in ("JointPositionError", "JointLimitViolation"):
            raise DeskError(
                f"Recovery type '{recovery_type}' requires physical joint movement. "
                "Move the affected joints to their reference positions and confirm "
                "via the Desk interface. Consult the product manual for details."
            )
        content: dict[str, Any] = {"type": recovery_type}
        if recovery_type == "SafetyError":
            content["safetyErrors"] = recovery.get("safetyErrors", [])
        elif recovery_type == "SafeInputUnacknowledged":
            content["safeInputs"] = recovery.get("safeInputs", [])
        self.send_api_request("/api/safety/recovery:confirm", content=content)

    def reboot(self) -> None:
        """Initiate a system reboot. Closes all open connections."""
        self.send_api_request("/api/system:reboot")

    def shutdown(self) -> None:
        """Initiate a system shutdown."""
        self.send_api_request("/api/system:shutdown")

    @property
    def __auth_string(self):
        return base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
