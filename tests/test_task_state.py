from __future__ import annotations

import time

from app.services.task_manager import TaskManager


def test_task_manager_success_and_failure(tmp_path) -> None:
    manager = TaskManager(max_workers=4, history_file=tmp_path / "jobs_history.jsonl")

    def success(ctx):
        ctx.set_progress(1, 2, "half")
        time.sleep(0.05)
        ctx.set_progress(2, 2, "done")
        return {"ok": True}

    def failure(ctx):
        ctx.set_progress(1, 1, "about to fail")
        raise RuntimeError("boom")

    job_ok = manager.submit("extract", {}, success)
    job_fail = manager.submit("annotate", {}, failure)

    deadline = time.time() + 3
    ok_final = None
    fail_final = None
    while time.time() < deadline:
        ok_final = manager.get(job_ok.id)
        fail_final = manager.get(job_fail.id)
        if ok_final and fail_final and ok_final.status in {"succeeded", "failed"} and fail_final.status in {"succeeded", "failed"}:
            break
        time.sleep(0.05)

    assert ok_final is not None
    assert fail_final is not None
    assert ok_final.status == "succeeded"
    assert ok_final.result == {"ok": True}
    assert fail_final.status == "failed"
    assert "boom" in (fail_final.error or "")

    history = manager.list_history(limit=10)
    assert len(history) >= 2
    assert {item["status"] for item in history[:2]} <= {"succeeded", "failed"}


def test_task_manager_cancel_running_job(tmp_path) -> None:
    manager = TaskManager(max_workers=2, history_file=tmp_path / "jobs_history.jsonl")

    def long_run(ctx):
        for i in range(20):
            time.sleep(0.03)
            ctx.set_progress(i + 1, 20, f"tick-{i+1}")
        return {"ok": True}

    job = manager.submit("annotate", {}, long_run)
    time.sleep(0.08)
    cancel_state = manager.cancel(job.id)
    assert cancel_state.cancel_requested is True

    deadline = time.time() + 3
    final = None
    while time.time() < deadline:
        final = manager.get(job.id)
        if final and final.status in {"cancelled", "failed", "succeeded"}:
            break
        time.sleep(0.05)

    assert final is not None
    assert final.status == "cancelled"
