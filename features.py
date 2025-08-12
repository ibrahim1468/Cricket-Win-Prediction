import numpy as np

def feature_engineering(df):
    df = df.copy()
    df["Current Score"] = df["Innings Runs"]
    df["Wickets Remaining"] = 10 - df["Innings Wickets"]
    df["RRR"] = np.where(
        df["Balls Remaining"] > 0,
        df["Runs to Get"] / (df["Balls Remaining"] / 6),
        0,
    )
    return df