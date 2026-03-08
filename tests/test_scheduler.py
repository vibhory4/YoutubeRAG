import scheduler


def test_check_all_channels_no_channels(monkeypatch):
    class S:
        def get_tracked_channels(self):
            return []

    monkeypatch.setattr(scheduler, "PipelineState", lambda: S())
    scheduler.check_all_channels()


def test_check_all_channels_runs_pipeline_and_survives_errors(monkeypatch):
    class S:
        def get_tracked_channels(self):
            return ["@a", "@b"]

    calls = []

    def fake_run_pipeline(channel):
        calls.append(channel)
        if channel == "@b":
            raise RuntimeError("boom")
        return {"new_video_count": 1, "ingestion_stats": {"total_chunks": 2}, "errors": []}

    monkeypatch.setattr(scheduler, "PipelineState", lambda: S())
    monkeypatch.setattr(scheduler, "run_pipeline", fake_run_pipeline)

    scheduler.check_all_channels()
    assert calls == ["@a", "@b"]


def test_start_scheduler_configures_and_handles_keyboardinterrupt(monkeypatch):
    events = {"add_job": False, "start": False}

    class DummyScheduler:
        def add_job(self, func, trigger, id, name, replace_existing, next_run_time):
            events["add_job"] = True

        def shutdown(self, wait=False):
            return None

        def start(self):
            events["start"] = True
            raise KeyboardInterrupt()

    monkeypatch.setattr(scheduler, "BlockingScheduler", lambda: DummyScheduler())
    monkeypatch.setattr(scheduler.signal, "signal", lambda *_args, **_kwargs: None)

    scheduler.start_scheduler(interval_minutes=3)
    assert events["add_job"] is True
    assert events["start"] is True


# ──────────────────────────────────────────
# Coverage gap tests (P0)
# ──────────────────────────────────────────

def test_check_all_channels_logs_pipeline_errors(monkeypatch, caplog):
    import logging

    class S:
        def get_tracked_channels(self):
            return ["@a"]

    def fake_run_pipeline(channel):
        return {
            "new_video_count": 0,
            "ingestion_stats": {"total_chunks": 0},
            "errors": ["e1", "e2", "e3", "e4", "e5", "e6"],
        }

    monkeypatch.setattr(scheduler, "PipelineState", lambda: S())
    monkeypatch.setattr(scheduler, "run_pipeline", fake_run_pipeline)

    with caplog.at_level(logging.WARNING, logger="scheduler"):
        scheduler.check_all_channels()

    assert "e1" in caplog.text
    assert "e5" in caplog.text


def test_start_scheduler_signal_handler_invokes_shutdown(monkeypatch):
    import signal as _signal
    import pytest

    captured_handlers = {}
    shutdown_called = []

    class DummyScheduler:
        def add_job(self, func, trigger, id, name, replace_existing, next_run_time):
            pass

        def shutdown(self, wait=False):
            shutdown_called.append(True)

        def start(self):
            raise KeyboardInterrupt()

    def fake_signal(signum, handler):
        captured_handlers[signum] = handler

    monkeypatch.setattr(scheduler, "BlockingScheduler", lambda: DummyScheduler())
    monkeypatch.setattr(scheduler.signal, "signal", fake_signal)

    scheduler.start_scheduler(interval_minutes=1)

    handler = captured_handlers.get(_signal.SIGTERM)
    assert handler is not None

    with pytest.raises(SystemExit):
        handler(_signal.SIGTERM, None)

    assert shutdown_called


# ──────────────────────────────────────────
# Robustness tests (P2)
# ──────────────────────────────────────────

def test_check_all_channels_exception_both_channels_run(monkeypatch):
    class S:
        def get_tracked_channels(self):
            return ["@a", "@b"]

    calls = []

    def fake_run_pipeline(channel):
        calls.append(channel)
        raise RuntimeError("fail")

    monkeypatch.setattr(scheduler, "PipelineState", lambda: S())
    monkeypatch.setattr(scheduler, "run_pipeline", fake_run_pipeline)

    scheduler.check_all_channels()
    assert calls == ["@a", "@b"]
