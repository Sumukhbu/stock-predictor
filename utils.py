import numpy as np
import pandas as pd

def add_features(df):
    df = df.copy()
    # Ensure an AdjClose column (handle both 'Adj Close' and 'AdjClose')
    if 'Adj Close' in df.columns:
        df.rename(columns={'Adj Close': 'AdjClose'}, inplace=True)
    if 'AdjClose' not in df.columns and 'Close' in df.columns:
        df['AdjClose'] = df['Close']

    df['return'] = df['AdjClose'].pct_change().fillna(0)
    df['log_return'] = np.log1p(df['return'])
    df['MA5'] = df['AdjClose'].rolling(5).mean().bfill()
    df['MA10'] = df['AdjClose'].rolling(10).mean().bfill()
    df['MA20'] = df['AdjClose'].rolling(20).mean().bfill()
    df['EMA20'] = df['AdjClose'].ewm(span=20, adjust=False).mean()
    df['vol_10'] = df['log_return'].rolling(10).std().bfill()
    df['mom_10'] = df['AdjClose'] - df['MA10']

    # RSI 14
    delta = df['AdjClose'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(span=14).mean()
    roll_down = down.ewm(span=14).mean()
    rs = roll_up / (roll_down + 1e-8)
    df['RSI14'] = 100 - (100 / (1 + rs))

    # OBV
    obv = [0]
    for i in range(1, len(df)):
        if df['AdjClose'].iloc[i] > df['AdjClose'].iloc[i-1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df['AdjClose'].iloc[i] < df['AdjClose'].iloc[i-1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv

    # fill any remaining NaNs with backward-fill then forward-fill as final safeguard
    df = df.bfill().ffill()
    return df

def create_sequences(df_features, target_col='AdjClose', seq_len=60, pred_horizon=1):
    """
    df_features: DataFrame where target_col is included as one column (scaled or raw depending on caller)
    Returns sequences (num_samples, seq_len, n_features) and targets (num_samples,)
    """
    features = df_features.values
    targets = df_features[target_col].values
    seqs = []
    ys = []
    for i in range(seq_len, len(features) - pred_horizon + 1):
        seqs.append(features[i - seq_len:i])
        ys.append(targets[i + pred_horizon - 1])
    return np.array(seqs), np.array(ys)
