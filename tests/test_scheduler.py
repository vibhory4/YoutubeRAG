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
