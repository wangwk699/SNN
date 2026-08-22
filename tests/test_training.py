from snn2.training import format_runtime_hms


def test_format_runtime_hms() -> None:
    assert format_runtime_hms(30550.7217) == "08:29:10.7217"


def test_format_runtime_hms_does_not_wrap_after_24_hours() -> None:
    assert format_runtime_hms(90061.5) == "25:01:01.5000"


def test_format_runtime_hms_carries_fractional_second_rounding() -> None:
    assert format_runtime_hms(3599.99996) == "01:00:00.0000"
