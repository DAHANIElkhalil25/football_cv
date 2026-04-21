import numpy as np
import yaml

from src.data.converter import (
    _mot_row_to_yolo,
    _read_tracklet_class_map,
    convert_soccernet_to_yolo,
    validate_yolo_dataset,
)


def test_mot_row_to_yolo_basic() -> None:
    row = [1, 5, 100, 200, 50, 80, 1, 1, 0.9, 0]
    result = _mot_row_to_yolo(row=row, img_w=1920, img_h=1080)
    assert result is not None
    class_id, cx, cy, bw, bh = result
    assert class_id == 0
    assert 0.0 <= cx <= 1.0
    assert 0.0 <= cy <= 1.0
    assert 0.0 < bw <= 1.0
    assert 0.0 < bh <= 1.0


def test_validate_yolo_dataset_empty(tmp_path) -> None:
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "train").mkdir(parents=True)
    report = validate_yolo_dataset(str(tmp_path), splits=["train"])
    assert report["is_valid"] is True
    assert report["splits"]["train"]["images"] == 0


def test_read_tracklet_class_map(tmp_path) -> None:
    gameinfo = tmp_path / "gameinfo.ini"
    gameinfo.write_text(
        "\n".join(
            [
                "[Sequence]",
                "trackletID_1= player team left;10",
                "trackletID_2= goalkeeper team right;X",
                "trackletID_3= referee;main",
                "trackletID_4= ball;1",
            ]
        ),
        encoding="utf-8",
    )

    mapping = _read_tracklet_class_map(gameinfo)
    assert mapping[1] == 0
    assert mapping[2] == 1
    assert mapping[3] == 2
    assert mapping[4] == 3


def test_convert_uses_gameinfo_when_mot_class_is_unknown(tmp_path) -> None:
    src_root = tmp_path / "raw"
    seq_root = src_root / "train" / "SNMOT-001"
    (seq_root / "gt").mkdir(parents=True)
    (seq_root / "img1").mkdir(parents=True)

    (seq_root / "gameinfo.ini").write_text(
        "\n".join(
            [
                "[Sequence]",
                "trackletID_1= player team left;10",
            ]
        ),
        encoding="utf-8",
    )
    (seq_root / "gt" / "gt.txt").write_text(
        "1,1,100,200,50,80,1,-1,-1,-1\n",
        encoding="utf-8",
    )
    (seq_root / "img1" / "000001.jpg").write_bytes(b"fake-jpg")

    out_root = tmp_path / "processed"
    stats = convert_soccernet_to_yolo(
        soccernet_dir=str(src_root),
        output_dir=str(out_root),
        splits_mapping={"train": "train"},
        sampling_step=1,
        min_visibility=0.2,
        min_bbox_px=1,
    )

    assert stats["totals"]["labels_written"] == 1
    labels = sorted((out_root / "labels" / "train").glob("*.txt"))
    assert len(labels) == 1
    label_line = labels[0].read_text(encoding="utf-8").strip()
    assert label_line.startswith("0 ")


def test_convert_sets_yaml_val_to_test_when_no_val_split(tmp_path) -> None:
    src_root = tmp_path / "raw"
    train_seq = src_root / "train" / "SNMOT-001"
    test_seq = src_root / "test" / "SNMOT-002"

    for seq_root in [train_seq, test_seq]:
        (seq_root / "gt").mkdir(parents=True)
        (seq_root / "img1").mkdir(parents=True)
        (seq_root / "gameinfo.ini").write_text(
            "[Sequence]\ntrackletID_1= player team left;10\n",
            encoding="utf-8",
        )
        (seq_root / "gt" / "gt.txt").write_text(
            "1,1,100,200,50,80,1,-1,-1,-1\n",
            encoding="utf-8",
        )
        (seq_root / "img1" / "000001.jpg").write_bytes(b"fake-jpg")

    out_root = tmp_path / "processed"
    stats = convert_soccernet_to_yolo(
        soccernet_dir=str(src_root),
        output_dir=str(out_root),
        splits_mapping={"train": "train", "test": "test"},
        sampling_step=1,
        min_visibility=0.2,
        min_bbox_px=1,
    )

    assert stats["yolo_val_split"] == "test"
    yaml_path = out_root / "soccernet.yaml"
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert payload["val"] == "images/test"
