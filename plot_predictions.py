# plot_predictions.py
import pandas as pd
import matplotlib.pyplot as plt
import os

ticker = "GOOGL"
csv_file = f"{ticker}_predictions.csv"
df = pd.read_csv(csv_file)

if 'Date' in df.columns:
    x = pd.to_datetime(df['Date'])
else:
    x = df.index

plt.figure(figsize=(12,6))
plt.plot(x, df['AdjClose'], label='Actual AdjClose', linewidth=1.2)
mask = ~df['Predicted'].isna()
plt.plot(x[mask], df.loc[mask, 'Predicted'], linestyle='--', marker='o', label='Predicted', markersize=4)

plt.title(f"{ticker} — Actual vs Predicted")
plt.xlabel("Date" if 'Date' in df.columns else "Index")
plt.ylabel("Price")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
out_plot = f"{ticker}_predictions.png"
plt.savefig(out_plot, dpi=150)
print("Saved plot to", out_plot)
# plt.show()
