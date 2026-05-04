#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


def canonical(payload: dict) -> str:
    return "|".join(
        [
            str(int(payload["version"])),
            payload["customer"],
            payload["fingerprint"],
            payload["issued_at"],
            payload["license_id"],
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Sign an offline autoannotation license.dat file.")
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--customer", required=True)
    parser.add_argument("--license-id", required=True)
    parser.add_argument("--issued-at", default="")
    parser.add_argument("--private-key", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from datetime import date

    payload = {
        "version": 1,
        "customer": args.customer.strip(),
        "fingerprint": args.fingerprint.strip(),
        "issued_at": args.issued_at.strip() or date.today().isoformat(),
        "license_id": args.license_id.strip(),
    }
    private_key = serialization.load_pem_private_key(Path(args.private_key).read_bytes(), password=None)
    signature = private_key.sign(
        canonical(payload).encode("utf-8"),
        padding.PKCS1v15(),
        hashes.SHA256(),
    )
    payload["signature"] = base64.b64encode(signature).decode("ascii")
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
