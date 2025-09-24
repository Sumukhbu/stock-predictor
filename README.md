# Real-time Stock Prediction (RNN/LSTM) — Offline / API-free Prototype

This repository is a self-contained prototype that demonstrates a **near‑real‑time** stock prediction pipeline (Google / GOOGL) using an LSTM time-series model, live-ish data ingestion via `yfinance`, a FastAPI server with WebSocket broadcasting, and a small client demo page. No paid APIs or API keys required — everything uses free data or local simulation.

---

## What you get in this package
- `train.py` — training pipeline (fetches historical data from Yahoo via `yfinance`, creates engineered features, trains an LSTM, saves model & scaler)
- `app.py` — FastAPI server that:
  - loads the saved model + scaler
  - maintains a rolling buffer of latest features for the ticker
  - polls Yahoo Finance every `POLL_INTERVAL` seconds (configurable) and runs inference
  - exposes `/predict` REST endpoint and `/ws` WebSocket endpoint which pushes predictions to connected clients
- `utils.py` — shared preprocessing, feature engineering, dataset helpers
- `client_demo.html` — minimal web client demonstrating live predictions via WebSocket
- `requirements.txt`, `Dockerfile`, `start.sh`
- `model/` — folder where trained artifacts are saved (`best_model.pth`, `scaler.pkl` after training)

---

## Quick start (step-by-step)

> Tested on Linux/macOS/Windows WSL. Use Python 3.9+ (3.11 recommended).

1. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate    # Windows (PowerShell)
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train a quick demo model (small epochs for a fast run). This will produce `model/best_model.pth` and `model/scaler.pkl`:
   ```bash
   python train.py --ticker GOOGL --period 365 --interval 1d --epochs 10
   ```
   - `--period` supports `60d`, `365d`, `5y`, `max` etc. Use `interval 1m` only if you have access to minute-level data reliably.
   - For a more accurate model, increase `--epochs` to `50`–`200` and use a longer `--period`.

4. Start the server:
   ```bash
   python app.py
   ```
   By default FastAPI runs at `http://127.0.0.1:8000`.

5. Open the demo client in your browser:
   - Open `client_demo.html` in a browser (double-click or `open` it) and it will attempt to connect to `ws://127.0.0.1:8000/ws` and show live predictions.
   - Or visit the built-in docs: `http://127.0.0.1:8000/docs` for the REST endpoints.

6. Run faster / production:
   - Use Dockerfile included and build an image or run with `uvicorn` directly:
     ```bash
     uvicorn app:app --host 0.0.0.0 --port 8000 --reload
     ```

---

## Notes, tips & caveats
- This is a prototype and **not** a trading system. Do not trade live capital with this without extensive testing, slippage modelling, and risk controls.
- `yfinance` is used as a free data source. Minute-data availability may vary; for robust production streaming, use a streaming provider (Alpaca, IEX, Polygon, etc.).
- If you have limited time, set `--epochs 10` for a quick demo; increase epochs for better-quality models.
- The demo uses a sliding-window LSTM (seq_len=60 by default). You can tweak hyperparams at the top of `train.py` and `app.py`.

---

If you want, I can:
- Provide a ready-made trained model for immediate demo (smaller file), OR
- Extend the project with sentiment replay (offline), paper-trading adapters, monitoring hooks, or a Docker Compose development stack.

Enjoy! — Run the steps above to train and run the real-time prototype.
