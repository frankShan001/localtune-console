from scripts import wait_for_dashboard


class _Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def test_wait_until_ready_returns_true_for_healthy_endpoint(monkeypatch):
    monkeypatch.setattr(
        wait_for_dashboard.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(),
    )

    assert wait_for_dashboard.wait_until_ready(
        "http://127.0.0.1:6543/api/status",
        timeout=1,
        interval=0,
    )


def test_wait_until_ready_returns_false_after_timeout(monkeypatch):
    timestamps = iter((0.0, 0.0, 1.0))
    monkeypatch.setattr(
        wait_for_dashboard.time,
        "monotonic",
        lambda: next(timestamps),
    )
    monkeypatch.setattr(wait_for_dashboard.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        wait_for_dashboard.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("not ready")),
    )

    assert not wait_for_dashboard.wait_until_ready(
        "http://127.0.0.1:6543/api/status",
        timeout=1,
        interval=0,
    )
