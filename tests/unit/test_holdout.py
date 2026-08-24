from pathlib import Path

import pytest

from bio_ml_preflight.provenance import HoldoutLedger


def test_holdout_access_requires_reason_after_limit(tmp_path: Path) -> None:
    ledger = HoldoutLedger(tmp_path / "holdout.jsonl", maximum_accesses=1)
    ledger.record_access(actor="tester", purpose="final confirmation")
    with pytest.raises(PermissionError, match="explicit override reason"):
        ledger.record_access(actor="tester", purpose="another look")
    ledger.record_access(actor="tester", purpose="audited correction", override_reason="label fix")
    assert ledger.entries()[-1]["override"] is True
