import pytest
from pathlib import Path
import zipfile

from src.data.downloader import (
    _download_split_with_fallback,
    _normalize_tracking_layout,
    download_soccernet_tracking,
    import_soccernet_from_zip,
    initialize_soccernet_layout,
    normalize_requested_splits,
)


def test_download_invalid_split_raises() -> None:
    with pytest.raises(ValueError):
        download_soccernet_tracking(
            local_dir="./tmp",
            password="dummy",
            splits=["invalid"],
        )


def test_normalize_requested_splits_supports_val_alias() -> None:
    normalized = normalize_requested_splits(["train", "val", "validation", "test"])
    assert normalized == ["train", "test"]


def test_initialize_soccernet_layout(tmp_path: Path) -> None:
    target = tmp_path / "soccernet"
    stats = initialize_soccernet_layout(str(target))

    assert (target / "train").exists()
    assert (target / "test").exists()
    assert (target / "README_IMPORT.md").exists()
    assert stats["mode"] == "init"


def test_import_soccernet_from_zip_local(tmp_path: Path) -> None:
    archive = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("train/SNMOT001/img1/000001.jpg", "x")
        zf.writestr("train/SNMOT001/gt/gt.txt", "1,1,10,10,20,20,1,0,1,0\n")
        zf.writestr("test/SNMOT002/img1/000001.jpg", "x")
        zf.writestr("test/SNMOT002/gt/gt.txt", "1,1,10,10,20,20,1,0,1,0\n")

    out_dir = tmp_path / "out"
    stats = import_soccernet_from_zip(local_dir=str(out_dir), zip_path=str(archive))

    assert stats["mode"] == "zip"
    assert stats["status"] == "imported"
    assert (out_dir / "train" / "SNMOT001" / "gt" / "gt.txt").exists()
    assert stats["sequences_detected"] >= 1


def test_normalize_tracking_layout_moves_split(tmp_path: Path) -> None:
    nested = tmp_path / "tracking-2023" / "train" / "SNMOT001"
    (nested / "img1").mkdir(parents=True)
    (nested / "gt").mkdir(parents=True)
    (nested / "gt" / "gt.txt").write_text("1,1,1,1,1,1,1,0,1,0\n", encoding="utf-8")

    _normalize_tracking_layout(output_root=tmp_path, split="train")

    assert (tmp_path / "train" / "SNMOT001" / "gt" / "gt.txt").exists()


def test_download_split_with_fallback_uses_tracking_2023_without_data_type(tmp_path: Path) -> None:
    class _FakeDownloader:
        def __init__(self) -> None:
            self.calls = []

        def downloadDataTask(self, task, split, verbose=True, password="x", source="HuggingFace"):
            self.calls.append({"task": task, "split": split, "password": password, "source": source})

    fake = _FakeDownloader()
    out = _download_split_with_fallback(
        downloader=fake,
        split="train",
        include_images=True,
        output_root=tmp_path,
        password="secret",
    )

    assert out["status"] == "ok"
    assert out["strategy"] in {"downloadDataTask:tracking", "downloadDataTask:tracking-2023"}
    assert fake.calls[0]["task"] in {"tracking", "tracking-2023"}
    assert fake.calls[0]["split"] == ["train"]


def test_download_marks_failure_when_no_artifacts(tmp_path: Path, monkeypatch) -> None:
    class _DummyDownloader:
        pass

    def _fake_build_downloader(local_dir: str, password: str):
        return _DummyDownloader()

    def _fake_download_split_with_fallback(**kwargs):
        return {"strategy": "fake", "status": "ok"}

    def _fake_verify_tracking_credentials(password: str, timeout_sec: int = 20):
        return None

    monkeypatch.setattr("src.data.downloader._build_downloader", _fake_build_downloader)
    monkeypatch.setattr("src.data.downloader._download_split_with_fallback", _fake_download_split_with_fallback)
    monkeypatch.setattr("src.data.downloader._verify_tracking_credentials", _fake_verify_tracking_credentials)

    with pytest.raises(ConnectionError):
        download_soccernet_tracking(
            local_dir=str(tmp_path / "soccernet"),
            password="secret",
            splits=["train"],
            download_videos=False,
        )


def test_download_split_with_fallback_uses_direct_webdav(tmp_path: Path, monkeypatch) -> None:
    class _NoApiDownloader:
        pass

    def _fake_direct(output_root, split, password):
        split_dir = Path(output_root) / split / "SNMOT001" / "gt"
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "gt.txt").write_text("1,1,1,1,20,20,1,0,1,0\n", encoding="utf-8")
        return {"strategy": "direct-webdav:tracking-2023", "status": "ok"}

    monkeypatch.setattr("src.data.downloader._download_split_direct_webdav", _fake_direct)

    out = _download_split_with_fallback(
        downloader=_NoApiDownloader(),
        split="train",
        include_images=False,
        output_root=tmp_path,
        password="secret",
    )

    assert out["status"] == "ok"
    assert out["strategy"] == "direct-webdav:tracking-2023"