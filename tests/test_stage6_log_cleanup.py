from pathlib import Path


def test_executor_uses_event_log_and_latest_status_file():

    root = Path(__file__).resolve().parents[1]

    source = (
        root /
        "TradeAI" /
        "execution" /
        "executor.py"
    ).read_text(
        encoding="utf-8"
    )

    assert "runtime_account_status.json" in source
    assert "_last_account_log_snapshot" in source
    assert "INFO | ACCOUNT CHANGED" in source
    assert "os.replace(" in source
