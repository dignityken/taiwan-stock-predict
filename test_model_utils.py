import pandas as pd

from model_utils import prepare_training_data


def test_latest_row_is_predicted_not_trained():
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
        "close": [100.0, 101.0, 102.0],
        "max": [101.0, 103.0, 105.0],
        "feature": [1.0, 2.0, 3.0],
    })

    train, latest, prediction_date = prepare_training_data(
        df, ["feature"], "2026-01-07"
    )

    assert prediction_date == pd.Timestamp("2026-01-06")
    assert train["Date"].tolist() == list(pd.to_datetime(["2026-01-02", "2026-01-05"]))
    assert train["Target"].astype(int).tolist() == [1, 1]
    assert latest["Date"].tolist() == [pd.Timestamp("2026-01-06")]


def test_limit_up_proxy_target():
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
        "close": [100.0, 100.0, 100.0],
        "max": [101.0, 109.5, 105.0],
        "feature": [1.0, 2.0, 3.0],
    })

    train, _, _ = prepare_training_data(df, ["feature"], "2026-01-06", target_return=0.095)

    assert train["Target"].astype(int).tolist() == [1, 0]


if __name__ == "__main__":
    test_latest_row_is_predicted_not_trained()
    test_limit_up_proxy_target()
    print("ok")
