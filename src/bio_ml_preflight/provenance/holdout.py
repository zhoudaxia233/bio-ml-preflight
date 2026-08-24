from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class HoldoutLedger:
    def __init__(self, path: Path, maximum_accesses: int = 1) -> None:
        self.path = path
        self.maximum_accesses = maximum_accesses

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [
            json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line
        ]

    def record_access(
        self,
        *,
        actor: str,
        purpose: str,
        override_reason: str | None = None,
        case_fingerprint: str | None = None,
    ) -> None:
        count = sum(entry["event"] == "access" for entry in self.entries())
        if count >= self.maximum_accesses and not override_reason:
            raise PermissionError(
                f"Holdout access limit ({self.maximum_accesses}) reached; "
                "provide an explicit override reason"
            )
        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": "access",
            "actor": actor,
            "purpose": purpose,
            "access_number": count + 1,
            "override": count >= self.maximum_accesses,
            "override_reason": override_reason,
            "case_fingerprint": case_fingerprint,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
