from src.detection.trainer import YOLOTrainer


def test_resolve_device_auto_returns_string() -> None:
    device = YOLOTrainer._resolve_device("auto")
    assert isinstance(device, str)
    assert device in {"cpu", "0"}


def test_resolve_device_cpu() -> None:
    assert YOLOTrainer._resolve_device("cpu") == "cpu"