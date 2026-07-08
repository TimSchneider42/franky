"""
Unit tests for franky/desk.py using mocked HTTP connections and websockets.

Covers both the legacy Franka Desk API (DeskWebSession) and the current
v1 API (Desk), plus the Pilot button websocket logic shared via BaseDesk.
No robot or network access is required.
"""

from __future__ import annotations

import http.client
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

try:
    from franky import desk
except ImportError:
    # The compiled _franky extension is not required for desk.py; load it
    # directly so these tests also run in a plain source checkout.
    import importlib.util
    import sys

    _DESK_PATH = Path(__file__).resolve().parents[2] / "franky" / "desk.py"
    _spec = importlib.util.spec_from_file_location("franky_desk", _DESK_PATH)
    desk = importlib.util.module_from_spec(_spec)
    sys.modules["franky_desk"] = desk
    _spec.loader.exec_module(desk)


HOST = "robot.example.com"
USER = "frank"
PASSWORD = "s3cret"


# ---------------------------------------------------------------------------
# Mock infrastructure
# ---------------------------------------------------------------------------


@dataclass
class RecordedRequest:
    method: str
    path: str
    headers: Dict[str, str]
    body: Any


class MockResponse:
    def __init__(self, status=200, body=b"", reason="", headers=None):
        self.status = status
        self.reason = reason
        self.headers = headers or {}
        self._body = body

    def read(self) -> bytes:
        body, self._body = self._body, b""
        return body

    def getheader(self, name, default=None):
        for key, value in self.headers.items():
            if key.lower() == name.lower():
                return value
        return default


def _to_response(value) -> MockResponse:
    if isinstance(value, MockResponse):
        return value
    if isinstance(value, bytes):
        return MockResponse(body=value)
    if isinstance(value, (dict, list)):
        return MockResponse(
            body=json.dumps(value).encode(),
            headers={"Content-Type": "application/json"},
        )
    raise TypeError(f"Unsupported route value: {value!r}")


class MockConnection:
    """Stands in for http.client.HTTPSConnection.

    Routes map (method, target) to a response, where a response is bytes,
    a JSON-serializable dict/list, a MockResponse, a callable taking the
    RecordedRequest (which may raise), or a list of any of these that is
    consumed one element per request.
    """

    def __init__(self, routes: Dict[tuple, Any]):
        self.routes = dict(routes)
        self.requests: List[RecordedRequest] = []
        self.connected = False
        self.closed = False
        self._pending: Optional[RecordedRequest] = None

    def connect(self):
        self.connected = True

    def close(self):
        self.closed = True

    def request(self, method, target, headers=None, body=None):
        self._pending = RecordedRequest(method, target, dict(headers or {}), body)
        self.requests.append(self._pending)

    def getresponse(self) -> MockResponse:
        req = self._pending
        key = (req.method, req.path)
        if key not in self.routes:
            raise AssertionError(f"Unexpected request: {key}")
        value = self.routes[key]
        if isinstance(value, list):
            value = value.pop(0)
        if callable(value):
            value = value(req)
        return _to_response(value)

    def requests_to(self, target: str) -> List[RecordedRequest]:
        return [r for r in self.requests if r.path == target]


class FakeWebSocket:
    """Queue-backed replacement for a websockets sync client connection."""

    def __init__(self, messages=()):
        self.messages = list(messages)
        self.closed = False

    def recv(self, timeout=None):
        if not self.messages:
            raise TimeoutError
        item = self.messages.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        self.closed = True


class FakeTime:
    """Replacement for the time module where sleep() just advances the clock."""

    def __init__(self):
        self.now = 0.0
        self.sleeps: List[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float):
        self.sleeps.append(seconds)
        self.now += seconds


# ---------------------------------------------------------------------------
# Session factories
# ---------------------------------------------------------------------------

LOGIN_TOKEN = "legacy-auth-token"


def legacy_control_routes() -> Dict[tuple, Any]:
    return {
        ("POST", "/admin/api/control-token/request"): {"token": "ctl-token", "id": 42},
        ("GET", "/admin/api/system-status"): {
            "controlToken": {"activeToken": {"id": 42}}
        },
        ("DELETE", "/admin/api/control-token"): b"",
    }


def make_legacy(monkeypatch, routes: Optional[Dict[tuple, Any]] = None):
    conn = MockConnection(
        {
            ("POST", "/admin/api/login"): LOGIN_TOKEN.encode(),
            **(routes or {}),
        }
    )
    monkeypatch.setattr(desk, "HTTPSConnection", lambda *a, **kw: conn)
    return desk.DeskWebSession(HOST, USER, PASSWORD), conn


def v1_control_routes() -> Dict[tuple, Any]:
    return {
        ("POST", "/api/system/control-token:take"): {
            "token": "ctl-token",
            "tokenId": 7,
        },
        ("GET", "/api/system/control-token"): {"tokenId": 7},
        ("POST", "/api/system/control-token:release"): lambda req: MockResponse(
            status=204
        ),
    }


def make_v1(monkeypatch, routes: Optional[Dict[tuple, Any]] = None):
    conn = MockConnection(
        {
            ("GET", "/api/system"): {"status": "Idle"},
            **(routes or {}),
        }
    )
    monkeypatch.setattr(desk, "HTTPSConnection", lambda *a, **kw: conn)
    return desk.Desk(HOST, USER, PASSWORD), conn


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


@pytest.mark.timeout(20)
def test_encode_password_known_value():
    assert desk._encode_password(USER, PASSWORD) == (
        "MTU3LDE3MiwyNDksMTEyLDE3MiwxMTcsMTk0LDE1NSwyMjEsNjcsMTQ3LDEyMCwyNTQsOCwxMjIs\n"
        "NTgsMjMyLDExNSwyNDksNzUsNTMsMTg0LDExMSw3MywxNTQsMTQyLDI0Niw2MSwyOCwxMzUsMjA2\n"
        "LDc3\n"
    )


# Name-mangled access to BaseDesk.__parse_pilot_button_payload.
_parse_pilot_button_payload = desk.BaseDesk._BaseDesk__parse_pilot_button_payload


@pytest.mark.timeout(20)
def test_parse_pilot_button_payload_parses_events():
    events = _parse_pilot_button_payload('{"circle": true, "cross": false}')
    assert events == [
        desk.PilotButtonEvent(button=desk.PilotButton.CIRCLE, pressed=True),
        desk.PilotButtonEvent(button=desk.PilotButton.CROSS, pressed=False),
    ]


@pytest.mark.timeout(20)
def test_parse_pilot_button_payload_ignores_non_object():
    assert _parse_pilot_button_payload("[1, 2, 3]") == []
    assert _parse_pilot_button_payload('"heartbeat"') == []


@pytest.mark.timeout(20)
def test_parse_pilot_button_payload_unknown_button_raises():
    with pytest.raises(ValueError):
        _parse_pilot_button_payload('{"not-a-button": true}')


# ---------------------------------------------------------------------------
# DeskWebSession (legacy API)
# ---------------------------------------------------------------------------


class TestDeskWebSession:
    def test_open_logs_in_with_encoded_password(self, monkeypatch):
        session, conn = make_legacy(monkeypatch)
        session.open()

        assert session.is_open
        assert session.token == LOGIN_TOKEN
        assert conn.connected
        (login,) = conn.requests_to("/admin/api/login")
        assert login.method == "POST"
        assert login.headers["content-type"] == "application/json"
        payload = json.loads(login.body)
        assert payload == {
            "login": USER,
            "password": desk._encode_password(USER, PASSWORD),
        }

    def test_requests_carry_auth_cookie(self, monkeypatch):
        session, conn = make_legacy(monkeypatch, legacy_control_routes())
        session.open()
        session._get_system_status()

        (status_req,) = conn.requests_to("/admin/api/system-status")
        assert status_req.headers["Cookie"] == f"authorization={LOGIN_TOKEN}"

    def test_open_twice_raises(self, monkeypatch):
        session, _ = make_legacy(monkeypatch)
        session.open()
        with pytest.raises(RuntimeError, match="already open"):
            session.open()

    def test_close_when_not_open_raises(self, monkeypatch):
        session, _ = make_legacy(monkeypatch)
        with pytest.raises(RuntimeError, match="not open"):
            session.close()

    def test_context_manager_opens_and_closes(self, monkeypatch):
        session, conn = make_legacy(monkeypatch)
        with session as s:
            assert s is session
            assert session.is_open
        assert not session.is_open
        assert conn.closed

    def test_api_error_raises_franka_api_error(self, monkeypatch):
        session, _ = make_legacy(
            monkeypatch,
            {
                ("GET", "/admin/api/system-status"): lambda req: MockResponse(
                    status=403, reason="Forbidden", body=b"denied"
                )
            },
        )
        session.open()
        with pytest.raises(desk.FrankaAPIError) as exc_info:
            session._get_system_status()
        err = exc_info.value
        assert err.http_code == 403
        assert err.path == "/admin/api/system-status"
        assert err.message == "denied"

    def test_request_retries_on_remote_disconnected(self, monkeypatch):
        attempts = []

        def flaky(req):
            attempts.append(req)
            if len(attempts) < 3:
                raise http.client.RemoteDisconnected("connection dropped")
            return {"controlToken": {"activeToken": None}}

        session, _ = make_legacy(
            monkeypatch, {("GET", "/admin/api/system-status"): flaky}
        )
        session.open()
        assert session._get_system_status() == {"controlToken": {"activeToken": None}}
        assert len(attempts) == 3

    def test_request_gives_up_after_three_disconnects(self, monkeypatch):
        def always_fails(req):
            raise http.client.RemoteDisconnected("connection dropped")

        session, conn = make_legacy(
            monkeypatch, {("GET", "/admin/api/system-status"): always_fails}
        )
        session.open()
        with pytest.raises(http.client.RemoteDisconnected):
            session._get_system_status()
        assert len(conn.requests_to("/admin/api/system-status")) == 3

    def test_take_control_acquires_token(self, monkeypatch):
        session, conn = make_legacy(monkeypatch, legacy_control_routes())
        session.open()
        session.take_control()

        assert session.has_control
        (req,) = conn.requests_to("/admin/api/control-token/request")
        assert json.loads(req.body) == {"requestedBy": USER}

    def test_take_control_force_uses_force_endpoint(self, monkeypatch):
        routes = legacy_control_routes()
        routes[("POST", "/admin/api/control-token/request?force")] = routes.pop(
            ("POST", "/admin/api/control-token/request")
        )
        session, conn = make_legacy(monkeypatch, routes)
        session.open()
        session.take_control(force=True)

        assert len(conn.requests_to("/admin/api/control-token/request?force")) == 1
        assert session.has_control

    def test_take_control_timeout_raises(self, monkeypatch):
        routes = legacy_control_routes()
        routes[("GET", "/admin/api/system-status")] = {
            "controlToken": {"activeToken": None}
        }
        session, _ = make_legacy(monkeypatch, routes)
        monkeypatch.setattr(desk, "time", FakeTime())
        session.open()
        with pytest.raises(desk.TakeControlTimeoutError):
            session.take_control(wait_timeout=5.0)

    def test_control_request_without_control_raises(self, monkeypatch):
        session, _ = make_legacy(monkeypatch)
        session.open()
        with pytest.raises(RuntimeError, match="take_control"):
            session.unlock_brakes()

    def test_release_control_sends_delete_with_control_token(self, monkeypatch):
        session, conn = make_legacy(monkeypatch, legacy_control_routes())
        session.open()
        session.take_control()
        session.release_control()

        (release,) = conn.requests_to("/admin/api/control-token")
        assert release.method == "DELETE"
        assert release.headers["X-Control-Token"] == "ctl-token"
        assert json.loads(release.body) == {"token": "ctl-token"}
        assert not session.has_control

    def test_close_releases_control(self, monkeypatch):
        session, conn = make_legacy(monkeypatch, legacy_control_routes())
        session.open()
        session.take_control()
        session.close()

        assert len(conn.requests_to("/admin/api/control-token")) == 1
        assert conn.closed

    @pytest.mark.parametrize(
        "action, target",
        [
            ("unlock_brakes", "/desk/api/joints/unlock"),
            ("lock_brakes", "/desk/api/joints/lock"),
            ("set_mode_programming", "/desk/api/operating-mode/programming"),
            ("set_mode_execution", "/desk/api/operating-mode/execution"),
        ],
    )
    def test_control_endpoints(self, monkeypatch, action, target):
        routes = legacy_control_routes()
        routes[("POST", target)] = b""
        session, conn = make_legacy(monkeypatch, routes)
        session.open()
        session.take_control()
        getattr(session, action)()

        (req,) = conn.requests_to(target)
        assert req.headers["X-Control-Token"] == "ctl-token"

    def test_enable_fci_posts_base64_token(self, monkeypatch):
        routes = legacy_control_routes()
        routes[("POST", "/desk/api/system/fci")] = b""
        session, conn = make_legacy(monkeypatch, routes)
        session.open()
        session.take_control()
        session.enable_fci()

        (req,) = conn.requests_to("/desk/api/system/fci")
        assert req.body.startswith("token=")
        assert req.headers["content-type"] == "application/x-www-form-urlencoded"

    def test_start_task(self, monkeypatch):
        session, conn = make_legacy(monkeypatch, {("POST", "/desk/api/execution"): b""})
        session.open()
        session.start_task("my_task")

        (req,) = conn.requests_to("/desk/api/execution")
        assert req.body == "id=my_task"

    def test_execute_self_test(self, monkeypatch):
        routes = legacy_control_routes()
        routes[("GET", "/admin/api/system-status")] = [
            # take_control() poll
            {"controlToken": {"activeToken": {"id": 42}}},
            # execute_self_test: td2Timeout check
            {
                "controlToken": {"activeToken": {"id": 42}},
                "safety": {"recoverableErrors": {"td2Timeout": False}},
            },
            # polling until the self test finishes
            {
                "controlToken": {"activeToken": {"id": 42}},
                "safety": {"safetyControllerStatus": "SelfTest"},
            },
            {
                "controlToken": {"activeToken": {"id": 42}},
                "safety": {"safetyControllerStatus": "Idle"},
            },
        ]
        routes[("POST", "/admin/api/safety/td2-tests/execute")] = {
            "code": "SuccessResponse"
        }
        session, conn = make_legacy(monkeypatch, routes)
        monkeypatch.setattr(desk, "time", FakeTime())
        session.open()
        session.take_control()
        session.execute_self_test()

        assert len(conn.requests_to("/admin/api/safety/td2-tests/execute")) == 1


# ---------------------------------------------------------------------------
# Desk (v1 API)
# ---------------------------------------------------------------------------

V1_BASIC_AUTH = "Basic ZnJhbms6czNjcmV0"  # base64("frank:s3cret")


class TestDesk:
    def test_open_verifies_connection_with_basic_auth(self, monkeypatch):
        session, conn = make_v1(monkeypatch)
        session.open()

        assert session.is_open
        assert conn.connected
        (req,) = conn.requests_to("/api/system")
        assert req.method == "GET"
        assert req.headers["Authorization"] == V1_BASIC_AUTH

    def test_open_twice_raises(self, monkeypatch):
        session, _ = make_v1(monkeypatch)
        session.open()
        with pytest.raises(RuntimeError, match="already open"):
            session.open()

    def test_close_when_not_open_raises(self, monkeypatch):
        session, _ = make_v1(monkeypatch)
        with pytest.raises(RuntimeError, match="not open"):
            session.close()

    def test_context_manager_opens_and_closes(self, monkeypatch):
        session, conn = make_v1(monkeypatch)
        with session as s:
            assert s is session
            assert session.is_open
        assert not session.is_open
        assert conn.closed

    def test_api_error_raises_franka_api_error(self, monkeypatch):
        session, _ = make_v1(
            monkeypatch,
            {
                ("GET", "/api/fci"): lambda req: MockResponse(
                    status=401, reason="Unauthorized", body=b"bad credentials"
                )
            },
        )
        session.open()
        with pytest.raises(desk.FrankaAPIError) as exc_info:
            session.get_fci_status()
        err = exc_info.value
        assert err.http_code == 401
        assert err.path == "/api/fci"
        assert err.message == "bad credentials"

    def test_204_response_is_accepted(self, monkeypatch):
        session, _ = make_v1(
            monkeypatch,
            {("POST", "/api/system:reboot"): lambda req: MockResponse(status=204)},
        )
        session.open()
        session.reboot()  # must not raise

    def test_request_retries_on_remote_disconnected(self, monkeypatch):
        attempts = []

        def flaky(req):
            attempts.append(req)
            if len(attempts) < 3:
                raise http.client.RemoteDisconnected("connection dropped")
            return {"status": "Idle"}

        session, _ = make_v1(monkeypatch, {("GET", "/api/fci"): flaky})
        session.open()
        assert session.get_fci_status() == "Idle"
        assert len(attempts) == 3

    def test_request_gives_up_after_three_disconnects(self, monkeypatch):
        def always_fails(req):
            raise http.client.RemoteDisconnected("connection dropped")

        session, conn = make_v1(monkeypatch, {("GET", "/api/fci"): always_fails})
        session.open()
        with pytest.raises(http.client.RemoteDisconnected):
            session.get_fci_status()
        assert len(conn.requests_to("/api/fci")) == 3

    def test_take_control_acquires_token(self, monkeypatch):
        session, conn = make_v1(monkeypatch, v1_control_routes())
        session.open()
        session.take_control(wait_timeout=10.0)

        assert session.has_control
        (req,) = conn.requests_to("/api/system/control-token:take")
        assert req.headers["content-type"] == "application/json"
        assert json.loads(req.body) == {"owner": USER, "timeout": 10}

    def test_take_control_is_idempotent(self, monkeypatch):
        session, conn = make_v1(monkeypatch, v1_control_routes())
        session.open()
        session.take_control()
        session.take_control()

        assert len(conn.requests_to("/api/system/control-token:take")) == 1

    def test_has_control_false_without_token(self, monkeypatch):
        session, conn = make_v1(monkeypatch)
        session.open()
        assert not session.has_control
        # No token id yet, so no status request should have been made.
        assert conn.requests_to("/api/system/control-token") == []

    def test_has_control_false_when_token_id_differs(self, monkeypatch):
        routes = v1_control_routes()
        routes[("GET", "/api/system/control-token")] = {"tokenId": 999}
        session, _ = make_v1(monkeypatch, routes)
        session.open()
        session.take_control()
        assert not session.has_control

    def test_control_request_without_control_raises(self, monkeypatch):
        session, conn = make_v1(monkeypatch)
        session.open()
        with pytest.raises(RuntimeError, match="take_control"):
            session.unlock_brakes()
        assert conn.requests_to("/api/arm/joints:unlock") == []

    def test_release_control_sends_control_token(self, monkeypatch):
        session, conn = make_v1(monkeypatch, v1_control_routes())
        session.open()
        session.take_control()
        session.release_control()

        (release,) = conn.requests_to("/api/system/control-token:release")
        assert release.headers["X-Control-Token"] == "ctl-token"
        assert not session.has_control

    def test_release_control_without_token_is_noop(self, monkeypatch):
        session, conn = make_v1(monkeypatch)
        session.open()
        session.release_control()
        assert conn.requests_to("/api/system/control-token:release") == []

    def test_close_releases_control(self, monkeypatch):
        session, conn = make_v1(monkeypatch, v1_control_routes())
        session.open()
        session.take_control()
        session.close()

        assert len(conn.requests_to("/api/system/control-token:release")) == 1
        assert conn.closed

    @pytest.mark.parametrize(
        "action, target, body",
        [
            ("unlock_brakes", "/api/arm/joints:unlock", None),
            ("lock_brakes", "/api/arm/joints:lock", None),
            (
                "set_mode_programming",
                "/api/system/operating-mode:change",
                {"desiredOperatingMode": "Programming"},
            ),
            (
                "set_mode_execution",
                "/api/system/operating-mode:change",
                {"desiredOperatingMode": "Execution"},
            ),
            ("enable_fci", "/api/fci:activate", None),
            ("disable_fci", "/api/fci:deactivate", None),
        ],
    )
    def test_control_endpoints(self, monkeypatch, action, target, body):
        routes = v1_control_routes()
        routes[("POST", target)] = b""
        session, conn = make_v1(monkeypatch, routes)
        session.open()
        session.take_control()
        getattr(session, action)()

        (req,) = conn.requests_to(target)
        assert req.method == "POST"
        assert req.headers["X-Control-Token"] == "ctl-token"
        if body is None:
            assert req.body is None
        else:
            assert json.loads(req.body) == body

    def test_get_system_status(self, monkeypatch):
        session, _ = make_v1(monkeypatch)
        session.open()
        assert session._get_system_status() == {"status": "Idle"}

    def test_get_fci_status(self, monkeypatch):
        session, _ = make_v1(monkeypatch, {("GET", "/api/fci"): {"status": "Active"}})
        session.open()
        assert session.get_fci_status() == "Active"

    def test_execute_self_test_polls_until_done(self, monkeypatch):
        routes = v1_control_routes()
        routes[("POST", "/api/safety/self-tests:execute")] = b""
        routes[("GET", "/api/safety/self-tests")] = [
            {"status": "Running"},
            {"status": "Running"},
            {"status": "Passed"},
        ]
        session, conn = make_v1(monkeypatch, routes)
        monkeypatch.setattr(desk, "time", FakeTime())
        session.open()
        session.take_control()
        session.execute_self_test()

        assert len(conn.requests_to("/api/safety/self-tests:execute")) == 1
        assert len(conn.requests_to("/api/safety/self-tests")) == 3

    def test_get_recovery_status_none(self, monkeypatch):
        session, _ = make_v1(monkeypatch, {("GET", "/api/safety/recovery"): {}})
        session.open()
        assert session.get_recovery_status() is None

    def test_recover_noop_when_nothing_to_recover(self, monkeypatch):
        session, conn = make_v1(monkeypatch, {("GET", "/api/safety/recovery"): {}})
        session.open()
        session.recover()
        assert conn.requests_to("/api/safety/recovery:confirm") == []

    def test_recover_confirms_simple_error(self, monkeypatch):
        session, conn = make_v1(
            monkeypatch,
            {
                ("GET", "/api/safety/recovery"): {
                    "recovery": {"type": "SelfTestsElapsed"}
                },
                ("POST", "/api/safety/recovery:confirm"): b"",
            },
        )
        session.open()
        session.recover()

        (confirm,) = conn.requests_to("/api/safety/recovery:confirm")
        assert json.loads(confirm.body) == {"type": "SelfTestsElapsed"}

    def test_recover_forwards_safety_errors(self, monkeypatch):
        session, conn = make_v1(
            monkeypatch,
            {
                ("GET", "/api/safety/recovery"): {
                    "recovery": {
                        "type": "SafetyError",
                        "safetyErrors": ["E1", "E2"],
                    }
                },
                ("POST", "/api/safety/recovery:confirm"): b"",
            },
        )
        session.open()
        session.recover()

        (confirm,) = conn.requests_to("/api/safety/recovery:confirm")
        assert json.loads(confirm.body) == {
            "type": "SafetyError",
            "safetyErrors": ["E1", "E2"],
        }

    @pytest.mark.parametrize(
        "recovery_type", ["JointPositionError", "JointLimitViolation"]
    )
    def test_recover_physical_recovery_raises(self, monkeypatch, recovery_type):
        session, conn = make_v1(
            monkeypatch,
            {("GET", "/api/safety/recovery"): {"recovery": {"type": recovery_type}}},
        )
        session.open()
        with pytest.raises(desk.DeskError, match="physical joint movement"):
            session.recover()
        assert conn.requests_to("/api/safety/recovery:confirm") == []

    @pytest.mark.parametrize(
        "action, target",
        [("reboot", "/api/system:reboot"), ("shutdown", "/api/system:shutdown")],
    )
    def test_reboot_and_shutdown(self, monkeypatch, action, target):
        session, conn = make_v1(monkeypatch, {("POST", target): b""})
        session.open()
        getattr(session, action)()
        assert len(conn.requests_to(target)) == 1


# ---------------------------------------------------------------------------
# Pilot button polling (BaseDesk, shared by both session classes)
# ---------------------------------------------------------------------------


@dataclass
class WebsocketCapture:
    socket: FakeWebSocket
    connects: List[Dict[str, Any]] = field(default_factory=list)


def patch_websocket(monkeypatch, messages=()) -> WebsocketCapture:
    capture = WebsocketCapture(socket=FakeWebSocket(messages))

    def fake_connect(uri, ssl=None, additional_headers=None, open_timeout=None):
        capture.connects.append({"uri": uri, "headers": additional_headers})
        return capture.socket

    monkeypatch.setattr(desk.ws_sync, "connect", fake_connect)
    return capture


class TestPilotButtons:
    def test_poll_requires_open_session(self, monkeypatch):
        session, _ = make_v1(monkeypatch)
        with pytest.raises(RuntimeError, match="not open"):
            session.poll_buttons()

    def test_poll_returns_empty_on_timeout(self, monkeypatch):
        session, _ = make_v1(monkeypatch)
        patch_websocket(monkeypatch, messages=[])
        session.open()
        assert session.poll_buttons() == []

    def test_poll_parses_and_drains_buffered_events(self, monkeypatch):
        session, _ = make_v1(monkeypatch)
        patch_websocket(
            monkeypatch,
            messages=['{"circle": true}', '{"circle": false, "check": true}'],
        )
        session.open()
        events = session.poll_buttons()
        assert events == [
            desk.PilotButtonEvent(button=desk.PilotButton.CIRCLE, pressed=True),
            desk.PilotButtonEvent(button=desk.PilotButton.CIRCLE, pressed=False),
            desk.PilotButtonEvent(button=desk.PilotButton.CHECK, pressed=True),
        ]

    def test_poll_reuses_websocket(self, monkeypatch):
        session, _ = make_v1(monkeypatch)
        capture = patch_websocket(monkeypatch, messages=['{"up": true}'])
        session.open()
        session.poll_buttons()
        session.poll_buttons()
        assert len(capture.connects) == 1

    def test_websocket_uri_and_v1_auth_headers(self, monkeypatch):
        session, _ = make_v1(monkeypatch)
        capture = patch_websocket(monkeypatch)
        session.open()
        session.poll_buttons()
        (connect,) = capture.connects
        assert connect["uri"] == f"wss://{HOST}/desk/api/navigation/events"
        assert connect["headers"] == {"Authorization": V1_BASIC_AUTH}

    def test_websocket_legacy_auth_headers(self, monkeypatch):
        session, _ = make_legacy(monkeypatch)
        capture = patch_websocket(monkeypatch)
        session.open()
        session.poll_buttons()
        (connect,) = capture.connects
        assert connect["headers"] == {"authorization": LOGIN_TOKEN}

    def test_poll_closes_socket_on_error(self, monkeypatch):
        session, _ = make_v1(monkeypatch)
        capture = patch_websocket(monkeypatch, messages=[ConnectionError("boom")])
        session.open()
        with pytest.raises(ConnectionError):
            session.poll_buttons()
        assert capture.socket.closed
        # The socket reference must be dropped so the next poll reconnects.
        session.poll_buttons()
        assert len(capture.connects) == 2

    def test_close_closes_pilot_socket(self, monkeypatch):
        session, _ = make_v1(monkeypatch)
        capture = patch_websocket(monkeypatch, messages=['{"down": true}'])
        session.open()
        session.poll_buttons()
        session.close()
        assert capture.socket.closed
