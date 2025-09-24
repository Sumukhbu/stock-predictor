# train.py
"""
Robust stock sequence trainer/predictor.

Usage example:
    python train.py --ticker GOOGL --period 1y --interval 1d --epochs 10 --seq_len 10
"""

import argparse
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

def download_data(ticker: str, period: str, interval: str, auto_adjust: bool = True) -> pd.DataFrame:
    # Explicitly set auto_adjust to avoid FutureWarning
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=auto_adjust)
    if df.empty:
        raise ValueError(f"No data returned for ticker={ticker}, period={period}, interval={interval}")
    # Ensure an adjusted close column is present
    if 'Adj Close' not in df.columns and 'Close' in df.columns:
        df['Adj Close'] = df['Close']
    df = df[['Adj Close']].copy()
    df.rename(columns={'Adj Close': 'AdjClose'}, inplace=True)
    df.reset_index(inplace=True)  # keep Date as column
    return df

def create_sequences_from_array(arr: np.ndarray, seq_len: int):
    """Create X,y sequences from 1D array arr (assumed shape (n,))."""
    X, y = [], []
    for i in range(len(arr) - seq_len):
        X.append(arr[i:i+seq_len])
        y.append(arr[i+seq_len])
    return np.array(X), np.array(y)

def build_lstm(seq_len: int, n_features: int = 1) -> Sequential:
    model = Sequential()
    model.add(LSTM(64, input_shape=(seq_len, n_features), return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def safe_assign_predictions(df_combined: pd.DataFrame, seq_len: int, preds: np.ndarray, col_name: str = 'Predicted'):
    """
    Assign preds (1D array) into df_combined.loc[seq_len:, col_name]
    while handling length mismatches by trimming or padding with NaN.
    """
    if col_name not in df_combined.columns:
        df_combined[col_name] = np.nan

    expected_len = len(df_combined) - seq_len
    pred = np.asarray(preds).ravel()
    col_idx = df_combined.columns.get_loc(col_name)

    if len(pred) == expected_len:
        df_combined.iloc[seq_len:, col_idx] = pred
    elif len(pred) > expected_len:
        df_combined.iloc[seq_len:, col_idx] = pred[:expected_len]
        print(f"Warning: trimmed predictions from {len(pred)} to {expected_len} to fit df rows.")
    else:
        # fewer predictions than rows -> fill what we have and set the rest to NaN
        if len(pred) > 0:
            df_combined.iloc[seq_len:seq_len + len(pred), col_idx] = pred
        df_combined.iloc[seq_len + len(pred):, col_idx] = np.nan
        print(f"Warning: only {len(pred)} predictions; remaining {expected_len - len(pred)} rows set to NaN.")

def main(args):
    ticker = args.ticker
    period = args.period
    interval = args.interval
    epochs = args.epochs
    seq_len = args.seq_len
    batch_size = args.batch_size

    print(f"Fetching data for {ticker} {period} {interval} ...")
    df = download_data(ticker, period=period, interval=interval, auto_adjust=True)
    # df has columns: Date, AdjClose
    if len(df) < seq_len + 2:
        raise ValueError(f"Not enough data points ({len(df)}) for seq_len={seq_len}.")

    # train/test split (80/20)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    # scaler fit on train only
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_df[['AdjClose']]).reshape(-1)
    # full scaled series (train + test) using same scaler
    test_scaled = scaler.transform(test_df[['AdjClose']]).reshape(-1)
    full_scaled = np.concatenate([train_scaled, test_scaled], axis=0)

    # Build training sequences from train_scaled
    X_train, y_train = create_sequences_from_array(train_scaled, seq_len)
    if X_train.size == 0:
        raise ValueError("X_train is empty. Decrease seq_len or provide more data.")

    # Build test sequences so that there are exactly len(test_df) target rows.
    # We construct sequences starting from (split_idx - seq_len) up to (len(full_scaled) - seq_len - 1)
    test_start = max(0, len(train_scaled) - seq_len)  # usually split_idx - seq_len
    X_test = []
    y_test = []
    for i in range(test_start, len(full_scaled) - seq_len):
        seq = full_scaled[i:i+seq_len]
        target = full_scaled[i+seq_len]
        X_test.append(seq)
        y_test.append(target)
    X_test = np.array(X_test)
    y_test = np.array(y_test).reshape(-1, 1)

    # Make sure number of test sequences matches number of rows in test_df
    expected_test_rows = len(test_df)
    if len(X_test) != expected_test_rows:
        # This warns but we continue: the safe assignment will handle mismatches.
        print(f"Note: created {len(X_test)} test sequences but test dataframe has {expected_test_rows} rows.")

    # reshape to (samples, seq_len, features)
    X_train = X_train.reshape((X_train.shape[0], seq_len, 1))
    X_test = X_test.reshape((X_test.shape[0], seq_len, 1))

    model = build_lstm(seq_len, n_features=1)
    es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

    print(f"Training: X_train={X_train.shape}, y_train={y_train.shape} ...")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=1)

    # Predict
    if X_test.shape[0] > 0:
        preds_scaled = model.predict(X_test).ravel()
        preds_inv = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
    else:
        preds_inv = np.array([])

    # Prepare combined df for assignment and saving:
    # Combine tail(seq_len) of train_df + all test_df rows so that predictions line up from index seq_len onward
    tail_train = train_df.tail(seq_len).reset_index(drop=True)
    combined = pd.concat([tail_train, test_df], ignore_index=True).reset_index(drop=True)
    # Add column for predicted values
    combined['Predicted'] = np.nan
    # Assign predictions safely into combined starting at index seq_len
    safe_assign_predictions(combined, seq_len=seq_len, preds=preds_inv, col_name='Predicted')

    # For convenience, include original AdjClose and also an "IsTest" flag
    combined['IsTest'] = False
    combined.loc[seq_len:, 'IsTest'] = True

    out_csv = f"{ticker}_predictions.csv"
    combined.to_csv(out_csv, index=False)
    print(f"Saved combined (tail_train + test + predictions) to {out_csv}")

    # Print short summary
    n_pred = np.count_nonzero(~np.isnan(combined['Predicted'].values[seq_len:]))
    print(f"Predictions written: {n_pred} / {max(0, len(combined) - seq_len)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="GOOGL")
    parser.add_argument("--period", type=str, default="1y", help="yfinance period (e.g. 1y, 6mo)")
    parser.add_argument("--interval", type=str, default="1d", help="yfinance interval (e.g. 1d, 1h)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    main(args)
