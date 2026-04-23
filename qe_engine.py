#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf


# =========================
# 設定値
# =========================

MONTH_BIAS = {
    1: 3,
    2: 1,
    3: 3,
    4: 2,
    5: -4,
    6: -1,
    7: 1,
    8: -3,
    9: -2,
    10: 1,
    11: 4,
    12: 5,
}

TERM_BIAS = {
    "上旬": 3,
    "中旬": -1,
    "下旬": 2,
}

# pandas weekday: Mon=0 ... Fri=4
WEEKDAY_BIAS = {
    0: -1,  # Mon
    1: 0,   # Tue
    2: 1,   # Wed
    3: 1,   # Thu
    4: 1,   # Fri
}

MERCURY_PERIODS = [
    ("2026-02-26", "2026-03-20", -0.5),
    ("2026-06-29", "2026-07-23", -0.5),
    ("2026-10-24", "2026-11-13", -0.5),
]

STRONG_BUY_THRESHOLD = 0.575
BUY_THRESHOLD = 0.525
SELL_WATCH_THRESHOLD = 0.475

DEFAULT_VIX = 19.5
DEFAULT_VIX_THRESHOLD = 25.0
DEFAULT_BREADTH = 0.176
DEFAULT_BREADTH_THRESHOLD = 0.40

DEFAULT_SYMBOL = "^N225"
DEFAULT_START = "2006-01-04"
DEFAULT_MIN_SCORE = 7.0
DEFAULT_FUTURE_DAYS = 60


# =========================
# データクラス
# =========================

@dataclass
class Settings:
    base_date: pd.Timestamp
    current_vix: float = DEFAULT_VIX
    vix_threshold: float = DEFAULT_VIX_THRESHOLD
    current_breadth: float = DEFAULT_BREADTH
    breadth_threshold: float = DEFAULT_BREADTH_THRESHOLD
    strong_buy_threshold: float = STRONG_BUY_THRESHOLD
    buy_threshold: float = BUY_THRESHOLD
    sell_watch_threshold: float = SELL_WATCH_THRESHOLD
    min_score_filter: float = DEFAULT_MIN_SCORE
    future_days: int = DEFAULT_FUTURE_DAYS


# =========================
# 引数
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QE Engine")
    parser.add_argument("--prices", required=False, help="CSV path with Date,Close")
    parser.add_argument(
        "--events", required=False, help="Optional CSV path with Date,Event,Bias"
    )
    parser.add_argument("--outdir", default="out_qe", help="Output directory")
    parser.add_argument(
        "--base-date",
        default=None,
        help="Base date YYYY-MM-DD; default = last price date",
    )
    parser.add_argument(
        "--symbol", default=DEFAULT_SYMBOL, help="Ticker for yfinance (default: ^N225)"
    )
    parser.add_argument(
        "--start", default=DEFAULT_START, help="Start date for price download"
    )
    parser.add_argument("--vix", type=float, default=DEFAULT_VIX, help="Current VIX")
    parser.add_argument(
        "--vix-threshold",
        type=float,
        default=DEFAULT_VIX_THRESHOLD,
        help="VIX guard threshold",
    )
    parser.add_argument(
        "--breadth",
        type=float,
        default=DEFAULT_BREADTH,
        help="Current breadth ratio, e.g. 0.176",
    )
    parser.add_argument(
        "--breadth-threshold",
        type=float,
        default=DEFAULT_BREADTH_THRESHOLD,
        help="Breadth guard threshold",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help="Only trade when TotalScore >= this value",
    )
    parser.add_argument(
        "--future-days",
        type=int,
        default=DEFAULT_FUTURE_DAYS,
        help="How many future business days to export",
    )
    return parser.parse_args()


# =========================
# 補助関数
# =========================

def term_from_day(day: int) -> str:
    if day <= 10:
        return "上旬"
    if day <= 20:
        return "中旬"
    return "下旬"


def special_score(dt: pd.Timestamp) -> int:
    day = dt.day
    wd = dt.weekday()  # Mon=0 ... Fri=4

    if day <= 5:
        return 2

    if 25 <= day <= 31:
        return 1

    if wd == 4 and 8 <= day <= 14:
        return -2

    return 0


def mercury_score(dt: pd.Timestamp) -> float:
    for start_s, end_s, bias in MERCURY_PERIODS:
        start = pd.Timestamp(start_s)
        end = pd.Timestamp(end_s)
        if start <= dt <= end:
            return bias
    return 0.0


def signal_from_buy_prob(prob: float, settings: Settings) -> str:
    if pd.isna(prob):
        return ""

    if prob >= settings.strong_buy_threshold:
        return "強め買い"
    if prob >= settings.buy_threshold:
        return "買い候補"
    if prob <= settings.sell_watch_threshold:
        return "売り警戒"
    return "スルー"


def strategy_return_from_signal(signal: str, one_day_return: float) -> float:
    if pd.isna(one_day_return):
        return math.nan

    if signal in ("強め買い", "買い候補"):
        return one_day_return

    if signal == "売り警戒":
        return -one_day_return

    return 0.0


# =========================
# データ読込
# =========================

def load_prices_from_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"Date", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"prices.csv is missing columns: {sorted(missing)}")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    df = (
        df.dropna(subset=["Date", "Close"])
        .sort_values("Date")
        .drop_duplicates("Date")
        .reset_index(drop=True)
    )

    df["NextClose"] = df["Close"].shift(-1)
    return df


def _flatten_download_columns(raw: pd.DataFrame) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [
            "_".join([str(x) for x in col if str(x) != ""]).strip("_")
            for col in raw.columns.to_flat_index()
        ]
    return raw


def _pick_close_column(raw: pd.DataFrame, symbol: str) -> str:
    candidates = [
        "Close",
        "Adj Close",
        f"Close_{symbol}",
        f"Adj Close_{symbol}",
        "Close_^N225",
        "Adj Close_^N225",
    ]

    for col in candidates:
        if col in raw.columns:
            return col

    close_like = [c for c in raw.columns if "Close" in str(c)]
    if close_like:
        return close_like[0]

    raise ValueError(
        f"Close column not found in downloaded data. Columns: {list(raw.columns)}"
    )


def load_prices_from_yfinance(
    symbol: str, start: str, end: Optional[str] = None
) -> pd.DataFrame:
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")

    raw = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        actions=False,
    )

    if raw is None or raw.empty:
        raise ValueError(f"Price download failed for symbol: {symbol}")

    raw = _flatten_download_columns(raw).reset_index()

    if "Date" not in raw.columns:
        raw = raw.rename(columns={raw.columns[0]: "Date"})

    close_col = _pick_close_column(raw, symbol)

    df = raw[["Date", close_col]].copy()
    df = df.rename(columns={close_col: "Close"})
    df["Date"] = pd.to_datetime(df["Date"])
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    df = (
        df.dropna(subset=["Date", "Close"])
        .sort_values("Date")
        .drop_duplicates("Date")
        .reset_index(drop=True)
    )

    df["NextClose"] = df["Close"].shift(-1)
    return df


def load_prices(path: Optional[str], symbol: str, start: str) -> pd.DataFrame:
    if path:
        return load_prices_from_csv(path)
    return load_prices_from_yfinance(symbol=symbol, start=start)


def load_events(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["Date", "Event", "Bias"])

    df = pd.read_csv(path)
    required = {"Date", "Event", "Bias"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"events.csv is missing columns: {sorted(missing)}")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Bias"] = pd.to_numeric(df["Bias"], errors="coerce").fillna(0.0)

    return df[["Date", "Event", "Bias"]].sort_values("Date").reset_index(drop=True)


# =========================
# モデル構築
# =========================

def build_daily_model(
    prices: pd.DataFrame, events: pd.DataFrame, settings: Settings
) -> pd.DataFrame:
    df = prices.copy()

    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["WeekdayNo"] = df["Date"].dt.weekday
    df["Weekday"] = (
        df["Date"]
        .dt.day_name()
        .map(
            {
                "Monday": "Mon",
                "Tuesday": "Tue",
                "Wednesday": "Wed",
                "Thursday": "Thu",
                "Friday": "Fri",
            }
        )
        .fillna("")
    )
    df["Term"] = df["Day"].map(term_from_day)

    df["MonthScore"] = df["Month"].map(MONTH_BIAS).fillna(0)
    df["TermScore"] = df["Term"].map(TERM_BIAS).fillna(0)
    df["WeekdayScore"] = df["WeekdayNo"].map(WEEKDAY_BIAS).fillna(0)
    df["SpecialScore"] = df["Date"].map(special_score)
    df["MercuryScore"] = df["Date"].map(mercury_score)

    if len(events):
        event_agg = events.groupby("Date", as_index=False).agg(
            Event=("Event", lambda s: " / ".join(map(str, s))),
            EventScore=("Bias", "sum"),
        )
        df = df.merge(event_agg, how="left", on="Date")
    else:
        df["Event"] = ""
        df["EventScore"] = 0.0

    df["Event"] = df["Event"].fillna("")
    df["EventScore"] = df["EventScore"].fillna(0.0)

    df["VIXGuard"] = -1.0 if settings.current_vix >= settings.vix_threshold else 0.0
    df["BreadthGuard"] = (
        -1.0 if settings.current_breadth < settings.breadth_threshold else 0.0
    )

    df["TotalScore"] = (
        df["MonthScore"]
        + df["TermScore"]
        + df["WeekdayScore"]
        + df["SpecialScore"]
        + df["EventScore"]
        + df["MercuryScore"]
        + df["VIXGuard"]
        + df["BreadthGuard"]
    )

    df["BuyProb"] = 0.50 + (df["TotalScore"] / 100.0)
    df["BuyProb"] = df["BuyProb"].clip(lower=0.0, upper=1.0)

    df["Signal"] = df["BuyProb"].map(lambda x: signal_from_buy_prob(x, settings))
    df["TradeFlag"] = df["TotalScore"] >= settings.min_score_filter

    df["1DReturn"] = (df["NextClose"] / df["Close"]) - 1.0
    df["StrategyReturn"] = [
        strategy_return_from_signal(sig, ret) if flag else 0.0
        for sig, ret, flag in zip(df["Signal"], df["1DReturn"], df["TradeFlag"])
    ]

    equity = []
    peak = []
    drawdown = []

    eq = 1.0
    pk = 1.0

    for ret in df["StrategyReturn"]:
        if pd.isna(ret):
            equity.append(math.nan)
            peak.append(math.nan)
            drawdown.append(math.nan)
            continue

        eq *= 1.0 + ret
        pk = max(pk, eq)
        dd = (eq / pk) - 1.0

        equity.append(eq)
        peak.append(pk)
        drawdown.append(dd)

    df["Equity"] = equity
    df["Peak"] = peak
    df["Drawdown"] = drawdown
    df["Mode"] = df["Date"].apply(
        lambda x: "Actual" if x <= settings.base_date else "Future"
    )

    df["Reason"] = (
        "月="
        + df["MonthScore"].astype(str)
        + " / 旬="
        + df["TermScore"].astype(str)
        + " / 曜日="
        + df["WeekdayScore"].astype(str)
        + " / 特殊="
        + df["SpecialScore"].astype(str)
        + " / イベント="
        + df["EventScore"].astype(str)
        + " / 水星逆行="
        + df["MercuryScore"].astype(str)
        + " / VIX="
        + df["VIXGuard"].astype(str)
        + " / Breadth="
        + df["BreadthGuard"].astype(str)
    )

    return df


def make_future_dates(base_date: pd.Timestamp, days: int) -> pd.DataFrame:
    future_dates = pd.bdate_range(start=base_date + pd.Timedelta(days=1), periods=days)
    return pd.DataFrame({"Date": future_dates})


def build_future_rows(
    base_date: pd.Timestamp,
    days: int,
    events: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    df = make_future_dates(base_date, days)

    df["Close"] = pd.NA
    df["NextClose"] = pd.NA

    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["WeekdayNo"] = df["Date"].dt.weekday
    df["Weekday"] = (
        df["Date"]
        .dt.day_name()
        .map(
            {
                "Monday": "Mon",
                "Tuesday": "Tue",
                "Wednesday": "Wed",
                "Thursday": "Thu",
                "Friday": "Fri",
            }
        )
        .fillna("")
    )
    df["Term"] = df["Day"].map(term_from_day)

    df["MonthScore"] = df["Month"].map(MONTH_BIAS).fillna(0)
    df["TermScore"] = df["Term"].map(TERM_BIAS).fillna(0)
    df["WeekdayScore"] = df["WeekdayNo"].map(WEEKDAY_BIAS).fillna(0)
    df["SpecialScore"] = df["Date"].map(special_score)
    df["MercuryScore"] = df["Date"].map(mercury_score)

    if len(events):
        event_agg = events.groupby("Date", as_index=False).agg(
            Event=("Event", lambda s: " / ".join(map(str, s))),
            EventScore=("Bias", "sum"),
        )
        df = df.merge(event_agg, how="left", on="Date")
    else:
        df["Event"] = ""
        df["EventScore"] = 0.0

    df["Event"] = df["Event"].fillna("")
    df["EventScore"] = df["EventScore"].fillna(0.0)

    df["VIXGuard"] = -1.0 if settings.current_vix >= settings.vix_threshold else 0.0
    df["BreadthGuard"] = (
        -1.0 if settings.current_breadth < settings.breadth_threshold else 0.0
    )

    df["TotalScore"] = (
        df["MonthScore"]
        + df["TermScore"]
        + df["WeekdayScore"]
        + df["SpecialScore"]
        + df["EventScore"]
        + df["MercuryScore"]
        + df["VIXGuard"]
        + df["BreadthGuard"]
    )

    df["BuyProb"] = 0.50 + (df["TotalScore"] / 100.0)
    df["BuyProb"] = df["BuyProb"].clip(lower=0.0, upper=1.0)

    df["Signal"] = df["BuyProb"].map(lambda x: signal_from_buy_prob(x, settings))
    df["TradeFlag"] = df["TotalScore"] >= settings.min_score_filter

    df["1DReturn"] = pd.NA
    df["StrategyReturn"] = pd.NA
    df["Equity"] = pd.NA
    df["Peak"] = pd.NA
    df["Drawdown"] = pd.NA
    df["Mode"] = "Future"

    df["Reason"] = (
        "月="
        + df["MonthScore"].astype(str)
        + " / 旬="
        + df["TermScore"].astype(str)
        + " / 曜日="
        + df["WeekdayScore"].astype(str)
        + " / 特殊="
        + df["SpecialScore"].astype(str)
        + " / イベント="
        + df["EventScore"].astype(str)
        + " / 水星逆行="
        + df["MercuryScore"].astype(str)
        + " / VIX="
        + df["VIXGuard"].astype(str)
        + " / Breadth="
        + df["BreadthGuard"].astype(str)
    )

    return df


def build_near_term(model: pd.DataFrame, base_date: pd.Timestamp) -> pd.DataFrame:
    month_start = pd.Timestamp(base_date.year, base_date.month, 1)
    month_after_next = month_start + pd.offsets.MonthBegin(2)
    month_after_next_end = month_after_next + pd.offsets.MonthEnd(0)

    near = model[
        (model["Date"] >= month_start) & (model["Date"] <= month_after_next_end)
    ].copy()
    near["Mercury"] = near["MercuryScore"].apply(lambda x: "逆行中" if x != 0 else "")

    cols = [
        "Date",
        "Weekday",
        "Mode",
        "TotalScore",
        "BuyProb",
        "Signal",
        "TradeFlag",
        "Event",
        "Mercury",
        "Close",
        "Reason",
    ]
    return near[cols].reset_index(drop=True)


def build_future(
    model: pd.DataFrame, base_date: pd.Timestamp, days: int = DEFAULT_FUTURE_DAYS
) -> pd.DataFrame:
    future = model[model["Date"] > base_date].copy()
    future = future.head(days)
    future["Mercury"] = future["MercuryScore"].apply(lambda x: "逆行中" if x != 0 else "")

    cols = [
        "Date",
        "Weekday",
        "Mode",
        "TotalScore",
        "BuyProb",
        "Signal",
        "TradeFlag",
        "Event",
        "Mercury",
        "Reason",
    ]
    return future[cols].reset_index(drop=True)


def build_summary(model: pd.DataFrame) -> pd.DataFrame:
    valid = model["StrategyReturn"].dropna()

    trade_mask = model["TradeFlag"] & model["Signal"].isin(
        ["強め買い", "買い候補", "売り警戒"]
    )
    wins = int((model.loc[trade_mask, "StrategyReturn"] > 0).sum())
    losses = int((model.loc[trade_mask, "StrategyReturn"] < 0).sum())
    trades = int(trade_mask.sum())

    sharpe_like = math.nan
    if len(valid) > 1 and valid.std(ddof=1) != 0:
        sharpe_like = (valid.mean() / valid.std(ddof=1)) * math.sqrt(252)

    summary = {
        "Trades": trades,
        "Wins": wins,
        "Losses": losses,
        "WinRate": (wins / (wins + losses)) if (wins + losses) else math.nan,
        "AvgReturn": valid.mean() if len(valid) else math.nan,
        "CumReturn": (
            (model["Equity"].dropna().iloc[-1] - 1.0)
            if model["Equity"].dropna().size
            else math.nan
        ),
        "MaxDD": (
            model["Drawdown"].min() if model["Drawdown"].notna().any() else math.nan
        ),
        "SharpeLike": sharpe_like,
    }

    return pd.DataFrame([summary])


# =========================
# 保存
# =========================

def save_outputs(
    model: pd.DataFrame,
    near: pd.DataFrame,
    future: pd.DataFrame,
    summary: pd.DataFrame,
    outdir: str | Path,
) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model.to_csv(outdir / "unified_model.csv", index=False, encoding="utf-8-sig")
    near.to_csv(outdir / "near_term.csv", index=False, encoding="utf-8-sig")
    future.to_csv(outdir / "future_signal.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(outdir / "summary.csv", index=False, encoding="utf-8-sig")


# =========================
# メイン
# =========================

def main() -> None:
    args = parse_args()

    prices = load_prices(path=args.prices, symbol=args.symbol, start=args.start)
    events = load_events(args.events)

    base_date = pd.Timestamp(args.base_date) if args.base_date else prices["Date"].max()

    settings = Settings(
        base_date=base_date,
        current_vix=args.vix,
        vix_threshold=args.vix_threshold,
        current_breadth=args.breadth,
        breadth_threshold=args.breadth_threshold,
        min_score_filter=args.min_score,
        future_days=args.future_days,
    )

    model = build_daily_model(prices, events, settings)
    future_rows = build_future_rows(
        settings.base_date, settings.future_days, events, settings
    )

    merged_for_lists = pd.concat([model, future_rows], ignore_index=True)

    near = build_near_term(merged_for_lists, settings.base_date)
    future = build_future(merged_for_lists, settings.base_date, settings.future_days)
    summary = build_summary(model)

    save_outputs(model, near, future, summary, args.outdir)

    print("Done.")
    print(f"Base date   : {settings.base_date.date()}")
    print(f"Rows        : {len(model)}")
    print(f"Min score   : {settings.min_score_filter}")
    print(f"Future days : {settings.future_days}")
    print(f"Trades      : {int(summary.loc[0, 'Trades'])}")

    if pd.notna(summary.loc[0, "WinRate"]):
        print(f"WinRate     : {summary.loc[0, 'WinRate']:.2%}")
    else:
        print("WinRate     : n/a")

    if pd.notna(summary.loc[0, "AvgReturn"]):
        print(f"AvgReturn   : {summary.loc[0, 'AvgReturn']:.4%}")
    else:
        print("AvgReturn   : n/a")

    if pd.notna(summary.loc[0, "CumReturn"]):
        print(f"CumReturn   : {summary.loc[0, 'CumReturn']:.4%}")
    else:
        print("CumReturn   : n/a")

    if pd.notna(summary.loc[0, "MaxDD"]):
        print(f"MaxDD       : {summary.loc[0, 'MaxDD']:.4%}")
    else:
        print("MaxDD       : n/a")

    if pd.notna(summary.loc[0, "SharpeLike"]):
        print(f"SharpeLike  : {summary.loc[0, 'SharpeLike']:.3f}")
    else:
        print("SharpeLike  : n/a")

    print(f"Output dir  : {Path(args.outdir).resolve()}")


if __name__ == "__main__":
    main()
