from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from app.services.imageset_service import ImageSetService
from app.services.work_ledger_service import WorkLedgerService


def test_assignment_rejects_admin_and_disabled_operator(app_state):
    admin = app_state.get_user("user_admin")
    op = app_state.create_user("op1", "secret1", "operator")
    disabled = app_state.create_user("op2", "secret2", "operator")
    app_state.update_user(disabled.id, enabled=False)
    seed = Path(app_state.users_index_file).parent / "seed.jpg"
    seed.write_bytes(b"not-an-image")
    imageset = app_state.create_imageset("set1", "test", [seed], creator_id=admin.id)

    try:
        app_state.set_imageset_assignees(imageset.id, [admin.id])
        raise AssertionError("admin should not be assignable")
    except ValueError:
        pass

    try:
        app_state.set_imageset_assignees(imageset.id, [disabled.id])
        raise AssertionError("disabled operator should not be assignable")
    except ValueError:
        pass

    updated = app_state.set_imageset_assignees(imageset.id, [op.id])
    assert updated.assignee_ids == [op.id]


def test_assignment_review_lifecycle_removes_accepted_task(app_state):
    admin = app_state.get_user("user_admin")
    op = app_state.create_user("review_op", "secret1", "operator")
    seed = Path(app_state.users_index_file).parent / "review_seed.jpg"
    seed.write_bytes(b"not-an-image")
    imageset = app_state.create_imageset("review_set", "test", [seed], creator_id=admin.id)

    app_state.set_imageset_assignees(imageset.id, [op.id])
    submitted = app_state.submit_imageset_assignment(imageset.id, op.id, "处理完成")
    assert submitted.assignee_states[op.id]["status"] == "submitted"

    accepted = app_state.review_imageset_assignment(imageset.id, op.id, approved=True, reviewer_id=admin.id)
    assert op.id not in accepted.assignee_ids
    assert op.id not in accepted.assignee_states
    assert accepted.assignee_history[-1]["status"] == "accepted"

    listed = ImageSetService.list_imagesets(app_state, user=admin)
    item = next(x for x in listed["items"] if x["imageset_id"] == imageset.id)
    assert item["assignee_history"][-1]["status"] == "accepted"
    assert item["assignee_history"][-1]["reviewer_username"] == admin.username

    app_state.set_imageset_assignees(imageset.id, [op.id])
    app_state.submit_imageset_assignment(imageset.id, op.id, "重新处理")
    rejected = app_state.review_imageset_assignment(imageset.id, op.id, approved=False, review_note="继续修", reviewer_id=admin.id)
    assert op.id in rejected.assignee_ids
    assert rejected.assignee_states[op.id]["status"] == "rejected"


def test_work_ledger_records_assignment_flow(app_client):
    _, app, _ = app_client
    app_state = app.state.app_state
    admin = app_state.get_user("user_admin")
    op = app_state.create_user("ledger_op", "secret1", "operator")
    seed = Path(app_state.users_index_file).parent / "ledger_seed.jpg"
    seed.write_bytes(b"not-an-image")
    imageset = app_state.create_imageset("ledger_set", "test", [seed], creator_id=admin.id)

    app_state.set_imageset_assignees(
        imageset.id,
        [op.id],
        actor_id=admin.id,
        actor_username=admin.username,
    )
    app_state.submit_imageset_assignment(imageset.id, op.id, "做完了")
    app_state.review_imageset_assignment(imageset.id, op.id, approved=True, reviewer_id=admin.id)

    ledger = WorkLedgerService.list_entries(
        state=app_state,
        tasks=app.state.task_manager,
        current_user=admin,
        limit=20,
    )
    actions = [item["action"] for item in ledger["items"]]
    assert "分派任务" in actions
    assert "提交验收" in actions
    assert "已验收" in actions
    accepted = next(item for item in ledger["items"] if item["action"] == "已验收")
    assert accepted["actor_username"] == admin.username
    assert accepted["target_username"] == op.username

    operator_ledger = WorkLedgerService.list_entries(
        state=app_state,
        tasks=app.state.task_manager,
        current_user=op,
        limit=20,
    )
    assert any(item["action"] == "已验收" for item in operator_ledger["items"])

    cleared = WorkLedgerService.clear_visible_entries(
        state=app_state,
        tasks=app.state.task_manager,
        current_user=admin,
    )
    assert cleared["hidden_rows"] >= 3
    after_clear = WorkLedgerService.list_entries(
        state=app_state,
        tasks=app.state.task_manager,
        current_user=admin,
        limit=20,
    )
    assert after_clear["items"] == []


def test_machine_id_env_fallback(monkeypatch):
    import app.core.machine_fingerprint as fp

    monkeypatch.setattr(fp.platform, "system", lambda: "Linux")
    monkeypatch.setattr(fp, "_linux_indicators", lambda: [])
    monkeypatch.setattr(fp, "_primary_mac", lambda: "")
    monkeypatch.setenv("AUTOANNOTATION_MACHINE_ID", "docker-box-001")

    assert fp.compute_fingerprint() == fp._stable_hash(["docker-box-001"])


def test_install_signed_license_writes_only_after_validation(tmp_path, monkeypatch):
    import app.core.config as config
    import app.core.license as license_module

    license_file = tmp_path / "system" / "license.dat"
    monkeypatch.setattr(config, "LICENSE_FILE", license_file)
    monkeypatch.setattr(config, "TRIAL_STATE_FILE", tmp_path / "system" / "trial_state.dat")
    monkeypatch.setattr(license_module, "compute_fingerprint", lambda: "fp_test")

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("utf-8")
    monkeypatch.setattr(license_module, "PUBLIC_KEY_PEM", public_pem)

    payload = {
        "version": 1,
        "customer": "客户A",
        "fingerprint": "fp_test",
        "issued_at": "2026-05-04",
        "license_id": "lic_test_001",
    }
    message = "|".join([str(payload["version"]), payload["customer"], payload["fingerprint"], payload["issued_at"], payload["license_id"]])
    payload["signature"] = base64.b64encode(
        private_key.sign(message.encode("utf-8"), padding.PKCS1v15(), hashes.SHA256())
    ).decode("ascii")

    status = license_module.install_license(json.dumps(payload).encode("utf-8"))
    assert status.mode == "licensed"
    assert license_file.exists()

    bad = dict(payload)
    bad["fingerprint"] = "wrong"
    with pytest.raises(Exception):
        license_module.install_license(json.dumps(bad).encode("utf-8"))
    assert json.loads(license_file.read_text(encoding="utf-8"))["fingerprint"] == "fp_test"
