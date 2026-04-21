from pathlib import Path

import yaml

from src.data.roboflow_merge import build_class_id_remap, merge_roboflow_with_soccernet


def _touch_file(path: Path, content: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_build_class_id_remap_with_aliases() -> None:
    source_names = {0: "person", 1: "ball", 2: "goalkeeper", 3: "tree"}
    remap = build_class_id_remap(source_names)
    assert remap == {0: 0, 1: 3, 2: 1}


def test_merge_roboflow_with_soccernet_train_only(tmp_path: Path) -> None:
    soccernet = tmp_path / "processed"
    _touch_file(soccernet / "images" / "train" / "sn_train_1.jpg")
    (soccernet / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (soccernet / "labels" / "train" / "sn_train_1.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    _touch_file(soccernet / "images" / "val" / "sn_val_1.jpg")
    (soccernet / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (soccernet / "labels" / "val" / "sn_val_1.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    _touch_file(soccernet / "images" / "test" / "sn_test_1.jpg")
    (soccernet / "labels" / "test").mkdir(parents=True, exist_ok=True)
    (soccernet / "labels" / "test" / "sn_test_1.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    (soccernet / "soccernet.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(soccernet),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"},
                "nc": 4,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    roboflow = tmp_path / "roboflow"
    (roboflow / "train" / "images").mkdir(parents=True)
    (roboflow / "train" / "labels").mkdir(parents=True)
    _touch_file(roboflow / "train" / "images" / "rf_1.jpg")
    (roboflow / "train" / "labels" / "rf_1.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.1 0.1\n",
        encoding="utf-8",
    )
    (roboflow / "data.yaml").write_text(
        yaml.safe_dump({"names": ["person", "ball"]}, sort_keys=False),
        encoding="utf-8",
    )

    merged = tmp_path / "processed_merged"
    stats = merge_roboflow_with_soccernet(
        soccernet_processed_dir=str(soccernet),
        roboflow_dataset_dir=str(roboflow),
        output_dir=str(merged),
        include_roboflow_splits=["train"],
    )

    assert stats["totals"]["roboflow_images_added"] == 1
    assert stats["totals"]["roboflow_labels_added"] == 1
    assert (merged / "images" / "val" / "sn_val_1.jpg").exists()
    assert (merged / "images" / "test" / "sn_test_1.jpg").exists()

    merged_label_files = sorted((merged / "labels" / "train").glob("rf_train_*.txt"))
    assert len(merged_label_files) == 1
    label_lines = merged_label_files[0].read_text(encoding="utf-8").strip().splitlines()
    assert label_lines[0].startswith("0 ")
    assert label_lines[1].startswith("3 ")

    merged_yaml = yaml.safe_load((merged / "soccernet_roboflow.yaml").read_text(encoding="utf-8"))
    assert merged_yaml["train"] == "images/train"
    assert merged_yaml["val"] == "images/val"
