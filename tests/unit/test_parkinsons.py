import hashlib
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from bio_ml_preflight.data import parkinsons
from bio_ml_preflight.data.parkinsons import load_parkinsons_telemonitoring


def _raw_frame() -> pd.DataFrame:
    rows = 4
    return pd.DataFrame(
        {
            "subject#": [1, 1, 2, 2],
            "age": [60, 60, 70, 70],
            "sex": [0, 0, 1, 1],
            "test_time": [0.0, 1.0, 0.0, 1.0],
            "motor_UPDRS": [10.0, 11.0, 20.0, 21.0],
            "total_UPDRS": [15.0, 16.0, 25.0, 26.0],
            "Jitter(%)": [0.1] * rows,
            "Jitter(Abs)": [0.2] * rows,
            "Jitter:RAP": [0.3] * rows,
            "Jitter:PPQ5": [0.4] * rows,
            "Jitter:DDP": [0.5] * rows,
            "Shimmer": [0.6] * rows,
            "Shimmer(dB)": [0.7] * rows,
            "Shimmer:APQ3": [0.8] * rows,
            "Shimmer:APQ5": [0.9] * rows,
            "Shimmer:APQ11": [1.0] * rows,
            "Shimmer:DDA": [1.1] * rows,
            "NHR": [1.2] * rows,
            "HNR": [1.3] * rows,
            "RPDE": [1.4] * rows,
            "DFA": [1.5] * rows,
            "PPE": [1.6] * rows,
        }
    )


def test_parkinsons_adapter_normalizes_records_and_verifies_cache(tmp_path: Path) -> None:
    frame, metadata = load_parkinsons_telemonitoring(tmp_path, loader=_raw_frame)

    assert frame["subject_id"].tolist() == ["1", "1", "2", "2"]
    assert frame["motor_updrs"].tolist() == [10.0, 11.0, 20.0, 21.0]
    assert metadata["row_count"] == 4
    assert metadata["participant_count"] == 2
    assert metadata["records_per_participant"] == {"min": 2, "median": 2.0, "max": 2}
    assert metadata["participant_time_proxy"]["unique"] == 4
    assert metadata["participant_time_proxy"]["records_in_repeated_groups"] == 0
    assert "linearly interpolated" in metadata["target_semantics"]

    cached, cached_metadata = load_parkinsons_telemonitoring(
        tmp_path, loader=lambda: (_ for _ in ()).throw(AssertionError())
    )
    pd.testing.assert_frame_equal(frame, cached)
    assert cached_metadata == metadata

    with pytest.raises(ValueError, match="injected test loader"):
        load_parkinsons_telemonitoring(tmp_path)

    frame.iloc[:1].to_parquet(tmp_path / "parkinsons_telemonitoring.parquet", index=False)
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_parkinsons_telemonitoring(tmp_path, loader=_raw_frame)


def test_parkinsons_adapter_downloads_and_verifies_pinned_zip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    buffer = BytesIO()
    with ZipFile(buffer, mode="w") as archive:
        archive.writestr("parkinsons_updrs.data", _raw_frame().to_csv(index=False))
    archive_bytes = buffer.getvalue()
    archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    request: dict[str, object] = {}

    def fake_urlopen(url: str, timeout: int) -> BytesIO:
        request.update(url=url, timeout=timeout)
        return BytesIO(archive_bytes)

    monkeypatch.setattr(parkinsons, "urlopen", fake_urlopen)
    monkeypatch.setattr(parkinsons, "UCI_PARKINSONS_SHA256", "0" * 64)

    with pytest.raises(ValueError, match="pinned Parkinsons source checksum mismatch"):
        load_parkinsons_telemonitoring(tmp_path / "mismatch")

    monkeypatch.setattr(parkinsons, "UCI_PARKINSONS_SHA256", archive_sha256)
    cache_dir = tmp_path / "valid"

    frame, metadata = load_parkinsons_telemonitoring(cache_dir)

    assert request == {"url": parkinsons.UCI_PARKINSONS_URL, "timeout": 60}
    assert len(frame) == 4
    assert metadata["source_mode"] == "pinned-official"
    assert metadata["raw_sha256"] == archive_sha256
    raw_path = cache_dir / "parkinsons+telemonitoring.zip"
    assert raw_path.read_bytes() == archive_bytes

    raw_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="source checksum mismatch"):
        load_parkinsons_telemonitoring(cache_dir)


def test_parkinsons_adapter_rejects_unknown_cached_source_mode(tmp_path: Path) -> None:
    load_parkinsons_telemonitoring(tmp_path, loader=_raw_frame)
    metadata_path = tmp_path / "source.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["source_mode"] = "unknown"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported cached Parkinsons source mode"):
        load_parkinsons_telemonitoring(tmp_path, loader=_raw_frame)
