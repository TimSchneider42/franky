import base64
import hashlib
import http.client
import json
import ssl
import time
import urllib.parse
from http.client import HTTPSConnection, HTTPResponse
from typing import Dict, Optional, Any, Literal


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


class RobotWebSession:
    def __init__(self, hostname: str, username: str, password: str):
        self.__hostname = hostname
        self.__username = username
        self.__password = password

        self.__client = None
        self.__token = None
        self.__control_token = None
        self.__control_token_id = None

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
        self.__client = HTTPSConnection(self.__hostname, timeout=timeout, context=ssl._create_unverified_context())
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

    @property
    def client(self) -> HTTPSConnection:
        return self.__client

    @property
    def token(self) -> str:
        return self.__token

    @property
    def is_open(self) -> bool:
        return self.__token is not None
