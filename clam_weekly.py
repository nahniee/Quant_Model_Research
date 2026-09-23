"""Weekly CLAM, rebuilt after the Signal_Validation review.

Changes from the quarterly clam_model.py:

Training windows are built one ticker at a time, so each window is 120 days of a single
stock, which is what the model sees when it makes a prediction (finding F3). Training data
stops at TRAIN_END, which leaves a real out-of-time period to test on (finding F4).

The horizon is 5 trading days instead of 65, and the input is 120 days instead of 252.
The target is a single number, the next 5-day log return of the adjusted close. The old
65 x 5 output was only ever used for one column.

Training uses the top-N stocks by market cap from the validation database (500 by
default) instead of 94 hand-picked large caps, so it matches the stocks the model is
applied to. Prices come from the DuckDB file. Features are adjusted for splits and
dividends, standardised on the training window and clipped at 5 standard deviations.

The architecture is otherwise the original: three causal Conv1D layers, three LSTM layers,
attention and a dense head. One caveat: the top-N list comes from the later snapshot, so
it still carries survivorship bias even though targets stop at TRAIN_END.
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Input, Layer, LayerNormalization, LSTM
from tensorflow.keras.models import Model

HERE = Path(__file__).resolve().parent
CONFIG = {
    "seq_length": 120,
    "horizon": 5,
    "stride": 5,                       # one training window per week per ticker
    "train_start": "2013-01-01",
    "train_end": "2021-12-31",         # F4: data after this date is kept for out-of-time testing
    "val_years": 2,
    "n_tickers": 500,
    "cnn_layers": [{"filters": 128, "kernel_size": 5}] * 3,
    "lstm_layers": [{"units": 256}] * 3,
    "dropout": 0.2,
    "batch_size": 256,
    "epochs": 40,
    "clip_sigma": 5.0,
    "target": "raw",                   # 'raw' 5-day log return | 'cs_demeaned' (minus that week's cross-sectional mean) | 'cs_rank' (percentile within week)
    "bars": "daily",                   # 'daily' (seq_length days, horizon days) or 'weekly' (52 weekly bars -> next week)
    "db_path": str(HERE.parent / "Signal_Validation" / "data" / "signal_validation.duckdb"),
    "artifacts": {"model": "clam_weekly_model.keras", "scaler": "clam_weekly_scaler.pkl", "meta": "clam_weekly_meta.json"},
}
FEATURES = ["open", "high", "low", "close", "volume"]


@tf.keras.utils.register_keras_serializable(package="clam")
class Attention(Layer):
    def build(self, input_shape):
        self.W = self.add_weight(name="att_w", shape=(input_shape[-1], 1), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_b", shape=(input_shape[1], 1), initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        alpha = K.softmax(K.squeeze(e, axis=-1))
        return K.sum(x * K.expand_dims(alpha, axis=-1), axis=1)


def directional_accuracy(y_true, y_pred):
    return K.mean(tf.cast(K.equal(K.sign(y_true), K.sign(y_pred)), tf.float32))


def create_model(config: dict = CONFIG) -> Model:
    inp = Input(shape=(config["seq_length"], len(FEATURES)))
    x = inp
    for p in config["cnn_layers"]:
        x = Conv1D(p["filters"], p["kernel_size"], padding="causal", activation="relu")(x)
        x = LayerNormalization()(x)
        x = Dropout(config["dropout"])(x)
    for p in config["lstm_layers"]:
        x = LSTM(p["units"], return_sequences=True)(x)
        x = LayerNormalization()(x)
        x = Dropout(config["dropout"])(x)
    ctx = Attention()(x)
    h = Dense(64, activation="relu")(ctx)
    out = Dense(1, name="fwd_logret")(h)
    m = Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=3e-4), loss="huber", metrics=[directional_accuracy])
    return m


def load_prices(config: dict = CONFIG) -> pd.DataFrame:
    """Long frame (date, ticker, open..volume) for the top-N universe, adjusted for splits/dividends."""
    import duckdb
    con = duckdb.connect(config["db_path"], read_only=True)
    df = con.execute("""
        SELECT p.date, p.ticker, p.open, p.high, p.low, p.close, p.adj_close, p.volume
        FROM prices p JOIN universe u USING (ticker)
        WHERE u.rank_by_mcap <= $n AND p.date >= $start AND p.close > 0 AND p.adj_close > 0
        ORDER BY p.ticker, p.date""", {"n": config["n_tickers"], "start": config["train_start"]}).df()
    con.close()
    f = df.adj_close / df.close
    for c in ("open", "high", "low", "close"):
        df[c] = df[c] * f
    return df.drop(columns="adj_close")


def weekly_bars(g: pd.DataFrame) -> pd.DataFrame:
    """Daily rows -> one OHLCV bar per trading week, labelled by the week's last trading day
    (= the rebalance date used by the validator)."""
    wk = g.date.dt.to_period("W-FRI")
    b = g.groupby(wk).agg(date=("date", "last"), open=("open", "first"), high=("high", "max"),
                          low=("low", "min"), close=("close", "last"), volume=("volume", "sum"))
    return b.reset_index(drop=True)


def features_for_ticker(g: pd.DataFrame, bars: str = "daily") -> pd.DataFrame:
    """dlog O/H/L/C and dlog1p volume, indexed by date (first row dropped)."""
    if bars == "weekly":
        g = weekly_bars(g)
    out = pd.DataFrame(index=g.date)
    for c in ("open", "high", "low", "close"):
        out[c] = np.log(g[c].values)
    out["volume"] = np.log1p(g["volume"].values)
    return out.diff().dropna()


def window_ends(T: int, seq_len: int, horizon: int, stride: int) -> np.ndarray:
    """Positions t at which a window [t-seq_len+1, t] ends and t+horizon is still observed."""
    return np.arange(seq_len - 1, T - horizon, stride)


def make_windows(feat: np.ndarray, ends: np.ndarray, seq_len: int) -> np.ndarray:
    """(n, seq_len, 5) array of windows ending at each position in `ends`."""
    return np.stack([feat[t - seq_len + 1: t + 1] for t in ends]).astype(np.float32)


def forward_logret(dlog_close: np.ndarray, ends: np.ndarray, horizon: int) -> np.ndarray:
    """Target: sum of the next `horizon` daily log returns after each window end."""
    cs = np.concatenate([[0.0], np.cumsum(dlog_close)])
    return (cs[ends + 1 + horizon] - cs[ends + 1]).astype(np.float32)


def build_dataset(config: dict = CONFIG):
    prices = load_prices(config)
    prices = prices[prices.date <= pd.Timestamp(config["train_end"])]
    split = pd.Timestamp(config["train_end"]) - pd.DateOffset(years=config["val_years"])
    feats = {t: features_for_ticker(g, config["bars"]) for t, g in prices.groupby("ticker", sort=False)}
    scaler = StandardScaler().fit(pd.concat([f[f.index < split] for f in feats.values()]).values)

    parts = []
    for t, f in feats.items():
        ends = window_ends(len(f), config["seq_length"], config["horizon"], config["stride"])
        if len(ends) == 0:
            continue
        z = np.clip(scaler.transform(f.values), -config["clip_sigma"], config["clip_sigma"])
        parts.append((make_windows(z, ends, config["seq_length"]),
                      forward_logret(f["close"].values, ends, config["horizon"]),   # raw log-return units
                      f.index[ends].values))                                         # window end date
    X = np.concatenate([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts]); end_dates = np.concatenate([p[2] for p in parts])
    if config["target"] == "cs_demeaned":
        y = y - pd.Series(y).groupby(end_dates).transform("mean").values
    elif config["target"] == "cs_rank":
        # percentile rank within each week, centred on 0: the model learns "where in this week's
        # cross-section will this stock land", which is all a top-N ranking ever uses
        y = (pd.Series(y).groupby(end_dates).rank(pct=True).values - 0.5).astype(np.float32)
    is_val = end_dates >= np.datetime64(split)
    Xtr, ytr, Xva, yva = X[~is_val], y[~is_val], X[is_val], y[is_val]
    y_scale = float(ytr.std())
    return Xtr, ytr / y_scale, Xva, yva / y_scale, scaler, {"y_scale": y_scale, "n_tickers": len(feats), "split": str(split.date())}


def artifact_dir(config: dict = CONFIG, root: Path = HERE) -> Path:
    tag = config["target"] + ("_wbars" if config["bars"] == "weekly" else "") + ("_small" if config["lstm_layers"][0]["units"] < 256 else "")
    return root / "clam_weekly" / tag


def main(config: dict = CONFIG, out_dir: Path | None = None) -> Model:
    out_dir = out_dir or artifact_dir(config)
    print(f"CLAM weekly v2: seq={config['seq_length']} horizon={config['horizon']} "
          f"universe=top{config['n_tickers']} train<= {config['train_end']}", flush=True)
    Xtr, ytr, Xva, yva, scaler, meta = build_dataset(config)
    print(f"train {Xtr.shape}  val {Xva.shape}  tickers {meta['n_tickers']}  val from {meta['split']}", flush=True)

    model = create_model(config)
    model.summary()
    cb = [EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
          ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-6, verbose=1)]
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=config["epochs"],
                     batch_size=config["batch_size"], callbacks=cb, verbose=2)

    pred = model.predict(Xva, batch_size=1024, verbose=0).ravel()
    ic = float(pd.Series(pred).corr(pd.Series(yva), method="spearman"))
    da = float(np.mean(np.sign(pred) == np.sign(yva)))
    print(f"validation: rank IC={ic:.4f}  directional accuracy={da:.4f}", flush=True)

    a = config["artifacts"]
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / a["model"])
    joblib.dump(scaler, out_dir / a["scaler"])
    (out_dir / a["meta"]).write_text(json.dumps({**{k: v for k, v in config.items() if k != "artifacts"},
                                                 **meta, "val_rank_ic": ic, "val_dir_acc": da,
                                                 "epochs_run": len(hist.history["loss"]),
                                                 "history": {k: [float(x) for x in v] for k, v in hist.history.items()}}, indent=2))
    print(f"saved {a['model']}, {a['scaler']}, {a['meta']}", flush=True)
    return model


def load_artifacts(dir_: Path | None = None, config: dict = CONFIG):
    a = config["artifacts"]
    dir_ = dir_ or artifact_dir(config)
    model = tf.keras.models.load_model(dir_ / a["model"], custom_objects={"Attention": Attention,
                                                                          "directional_accuracy": directional_accuracy})
    return model, joblib.load(dir_ / a["scaler"]), json.loads((dir_ / a["meta"]).read_text())


if __name__ == "__main__":
    import sys
    cfg = dict(CONFIG)
    if len(sys.argv) > 1: cfg["target"] = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "weekly": cfg.update(bars="weekly", seq_length=52, horizon=1, stride=1)
    if len(sys.argv) > 2 and sys.argv[2] == "small": cfg.update(lstm_layers=[{"units": 128}] * 2, cnn_layers=[{"filters": 64, "kernel_size": 5}] * 2, dropout=0.3)
    main(cfg)
