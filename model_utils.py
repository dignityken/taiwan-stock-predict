import pandas as pd


def prepare_training_data(df, features, requested_date, target_return=0.015):
    df = df.sort_values("Date").reset_index(drop=True).copy()
    available = df.loc[df["Date"] <= pd.Timestamp(requested_date).normalize(), "Date"]
    if available.empty:
        return df.iloc[0:0], df.iloc[0:0], None

    prediction_date = available.max()
    next_high = df["max"].shift(-1)
    df["Target"] = ((next_high - df["close"]) / df["close"] >= target_return).astype("Int64")
    df.loc[next_high.isna(), "Target"] = pd.NA

    train_df = df[df["Date"] < prediction_date].dropna(subset=features + ["Target"]).copy()
    today_row = df[df["Date"] == prediction_date].copy()
    return train_df, today_row, prediction_date
