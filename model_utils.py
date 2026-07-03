import pandas as pd


def prepare_training_data(df, features, requested_date, target_return=0.006,
                          target_mode="open_close"):
    df = df.sort_values("Date").reset_index(drop=True).copy()
    available = df.loc[df["Date"] <= pd.Timestamp(requested_date).normalize(), "Date"]
    if available.empty:
        return df.iloc[0:0], df.iloc[0:0], None

    prediction_date = available.max()
    if target_mode == "open_close":
        # 可交易目標：次日開盤買、收盤賣，報酬需覆蓋來回成本（約 0.6%）
        next_return = (df["close"].shift(-1) - df["open"].shift(-1)) / df["open"].shift(-1)
    else:  # "high"：次日最高價相對今日收盤（漲停候選用）
        next_return = (df["max"].shift(-1) - df["close"]) / df["close"]
    df["Target"] = (next_return >= target_return).astype("Int64")
    df.loc[next_return.isna(), "Target"] = pd.NA

    train_df = df[df["Date"] < prediction_date].dropna(subset=features + ["Target"]).copy()
    today_row = df[df["Date"] == prediction_date].copy()
    return train_df, today_row, prediction_date
