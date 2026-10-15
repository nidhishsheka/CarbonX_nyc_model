"""Next-day credit price forecast for the trading screen.

This is the *market* model. It is unrelated to the emissions benchmark in benchmark.py.

Fixes versus the original:
  * Training and inference now use the same features. The old code trained on the
    previous day's price (lag 1) against the price two steps ahead, but predicted
    from today's price, so the model was applied to inputs it was never trained on.
  * The forecast now starts from the genuinely latest day. The old code dropped the
    last row (its target is unknown) and then treated the day before as "latest".
  * Too little history is reported instead of fitting 7 features to a handful of rows.
  * The naive "tomorrow = today" error is reported next to the model error, so the
    forecast is judged against the obvious alternative.
"""
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from db import get_db

FEATURES = [
    "avg_price", "moving_average", "price_change",
    "credits_traded", "trade_value", "active_listings", "credits_available",
]
MIN_USABLE_ROWS = 12  # rows with all features and a known next-day target


def get_price_prediction():
    conn = get_db()
    try:
        df = pd.read_sql_query(
            "SELECT trade_date, avg_price, credits_traded, trade_value, active_listings, "
            "credits_available FROM Market_History ORDER BY trade_date ASC", conn)
    finally:
        conn.close()

    if len(df) < 2:
        return {"available": False,
                "message": f"Need more historical data. Currently have {len(df)} trading periods."}

    df["moving_average"] = df["avg_price"].rolling(3).mean()
    df["price_change"] = df["avg_price"].pct_change() * 100
    df["target_price"] = df["avg_price"].shift(-1)

    latest = df.iloc[[-1]][FEATURES]          # today's features -> tomorrow's price
    train_df = df.dropna(subset=FEATURES + ["target_price"])

    if len(train_df) < MIN_USABLE_ROWS or latest.isna().any(axis=None):
        return {"available": False,
                "message": (f"Need at least {MIN_USABLE_ROWS} usable trading days to forecast; "
                            f"currently {len(train_df)}.")}

    split = int(len(train_df) * 0.8)
    X, y = train_df[FEATURES], train_df["target_price"]
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(X.iloc[:split], y.iloc[:split])
    mae = mean_absolute_error(y.iloc[split:], model.predict(X.iloc[split:]))
    naive_mae = mean_absolute_error(y.iloc[split:], X.iloc[split:]["avg_price"])

    model.fit(X, y)                           # refit on everything before forecasting
    predicted = float(model.predict(latest)[0])
    current = float(latest["avg_price"].iloc[0])
    return {
        "available": True,
        "predicted_price": round(predicted, 2),
        "current_price": round(current, 2),
        "mae": round(mae, 2),
        "naive_mae": round(naive_mae, 2),
    }


if __name__ == "__main__":
    print("\nAI Forecast:")
    print(get_price_prediction())
