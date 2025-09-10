# app_streamlit.py
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Price Channels & Bands", layout="wide")

# --- Named Watchlist (ticker -> display name) ---
WATCHLIST = {
    "1295.KL": "Public Bank",
    "NVDA": "Nvidia",
    "AAPL": "APPLE",
    "TSLA": "Tesla",
    "SPY": "S&P 500 ETF",
    "^IXIC": "NASDAQ",
    "^DJI": "DOW J",
    "^STI": "STI index",
    "URTH": "iShares MSCI World ETF",
    "C38U.SI": "CapitaLand Integrated Commercial Trust",
    "M44U.SI": "Mapletree Logistics Trust",
    "A17U.SI": "CapitaLand Ascendas REIT",
    "J69U.SI": "Frasers Centrepoint Trust",
    "ME8U.SI": "Mapletree Industrial Trust",
    "N2IU.SI": "Mapletree Pan Asia Commercial Trust",
    "D05.SI": "DBS",
    "C6L.SI": "SIA airline",
    "S63.SI": "ST Eng",
    "O39.SI": "OCBC",
    "BTC-USD": "Bitcoin price in USD dollars",
    "AWX.SI" : "AEM",
    "E28.SI" : "Frencken",
    "SIE.DE" : "Siemens",
    "CDNS"   : "Cadence",
    "AIQ"    : "AI etf - global x",
    "ARTY"   : "iShares Future AI & Tech ETF",
    "ROBT"   : "First Trust Nasdaq Artificial Intelligence and Robotics ETF",
    "BOTZ"   : "Global X Robotics & AI Thematic ETF",
    "IBOT"   : "Vaneck Robotics ETF",
    "IVES"   : "Dan IVES Wedbush AI Revolution ETF"
}

# ==== Daily OHLC + Overlays helpers ====
@st.cache_data(ttl=1800, show_spinner=False)
def _load_ohlc_from_yahoo_v2(
    ticker: str,
    start: str | None,
    end: str | None,
    interval: str = "1d",
) -> pd.DataFrame:
    interval = {"1d": "1d", "1wk": "1wk", "1mo": "1mo"}.get(str(interval), "1d")

    df = yf.download(
        ticker,
        period=None if start else "max",
        start=None if start is None else start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=False,
    )
    if df is None or df.empty:
        df = yf.Ticker(ticker).history(period="max", interval=interval, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()

    # Flatten possible MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        lv0 = df.columns.get_level_values(0)
        if ticker in lv0:
            df = df.xs(ticker, level=0, axis=1)
        else:
            df = df.droplevel(0, axis=1)

    # tz-naive index
    try:
        df.index = pd.to_datetime(df.index).tz_localize(None)
    except Exception:
        pass

    # Ensure Close exists
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    # Deduplicate column names
    if hasattr(df.columns, "duplicated"):
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # Helper: always return numeric Series by name
    def _get_series(frame: pd.DataFrame, name: str) -> pd.Series | None:
        if name not in frame.columns:
            return None
        sub = frame.loc[:, [name]]
        s = sub.iloc[:, 0] if sub.shape[1] == 1 else None
        if s is None:
            for i in range(sub.shape[1]):
                cand = sub.iloc[:, i]
                if cand.notna().any():
                    s = cand; break
            if s is None:
                s = sub.iloc[:, 0]
        return pd.to_numeric(s, errors="coerce")

    wanted = ["Open", "High", "Low", "Close", "Volume"]
    series_map = {name: s for name in wanted if (s := _get_series(df, name)) is not None}
    if not series_map:
        return pd.DataFrame()

    out = pd.concat(series_map, axis=1)

    # Drop rows missing OHLC; Volume optional
    ohlc_cols = [c for c in ["Open", "High", "Low", "Close"] if c in out.columns]
    if ohlc_cols:
        out = out.dropna(subset=ohlc_cols, how="any")
        out = out[out["Close"] > 0]

    return out

def bollinger_bands(series: pd.Series, period: int = 20, std_mult: float = 2.0):
    s = pd.to_numeric(series, errors="coerce")
    ma = s.rolling(period).mean()
    sd = s.rolling(period).std(ddof=0)
    upper = ma + std_mult * sd
    lower = ma - std_mult * sd
    return ma, upper, lower

def ichimoku(df_close: pd.Series, df_high: pd.Series, df_low: pd.Series,
             tenkan=9, kijun=26, senkou_b=52, displacement=26):
    tenkan_sen = (df_high.rolling(tenkan).max() + df_low.rolling(tenkan).min()) / 2.0
    kijun_sen  = (df_high.rolling(kijun).max()  + df_low.rolling(kijun).min())  / 2.0
    sen_a = ((tenkan_sen + kijun_sen) / 2.0).shift(displacement)
    sen_b = ((df_high.rolling(senkou_b).max() + df_low.rolling(senkou_b).min()) / 2.0).shift(displacement)
    chikou = df_close.shift(-displacement)
    return tenkan_sen, kijun_sen, sen_a, sen_b, chikou

def compute_cagr(dates, prices, window_days: int | None = None):
    """
    Start–end CAGR = (P_end / P_start) ** (365.25 / days) - 1
    If window_days is set, use only the last `window_days` of data.
    """
    s = pd.DataFrame({"dt": pd.to_datetime(dates), "px": pd.to_numeric(prices, errors="coerce")}).dropna()
    s = s.sort_values("dt")
    if window_days and window_days > 0:
        cutoff = s["dt"].iloc[-1] - pd.Timedelta(days=window_days)
        s = s[s["dt"] >= cutoff]
    if len(s) < 2:
        return None

    p0, p1 = float(s["px"].iloc[0]), float(s["px"].iloc[-1])
    d0, d1 = s["dt"].iloc[0], s["dt"].iloc[-1]
    days = (d1 - d0).days
    if days <= 0 or p0 <= 0 or p1 <= 0:
        return None
    return (p1 / p0) ** (365.25 / days) - 1

def _fit_log_line(dts: pd.Series, prices: pd.Series):
    dts = pd.to_datetime(dts)
    t = (dts - dts.iloc[0]).dt.total_seconds().values / 86400.0
    y = np.log(prices.astype(float).values)
    t0 = np.median(t)
    t_centered = t - t0
    b, a = np.polyfit(t_centered, y, 1)
    a_full = a - b * t0
    return a_full, b  # intercept (full), slope per day in log space

def compute_trend_cagr(dates, prices, window_days: int | None = None):
    """
    Regression-based CAGR using ALL points:
    Fit ln(price) ~ a + b * t_days over window; CAGR = exp(b * 365.25) - 1
    """
    s = pd.DataFrame({"dt": pd.to_datetime(dates), "px": pd.to_numeric(prices, errors="coerce")}).dropna()
    s = s.sort_values("dt")
    if window_days and window_days > 0 and len(s) >= 2:
        cutoff = s["dt"].iloc[-1] - pd.Timedelta(days=window_days)
        s = s[s["dt"] >= cutoff]
    if len(s) < 3 or (s["px"] <= 0).all():
        return None
    _, b = _fit_log_line(s["dt"], s["px"])
    return float(np.exp(b * 365.25) - 1.0)

def resolve_name(ticker: str) -> str:
    """Return a friendly display name for the ticker."""
    name = WATCHLIST.get(ticker)
    if name:
        return name
    try:
        info = yf.Ticker(ticker).info
        return info.get("longName") or info.get("shortName") or ticker
    except Exception:
        return ticker

# ---------- Yahoo helper ----------
def load_prices_from_yahoo(ticker: str, start: str | None, end: str | None,
                           interval: str = "1d", adjusted: bool = True) -> pd.DataFrame:
    data = yf.download(ticker, period="max" if start is None else None,
                       start=None if start is None else start,
                       end=end, interval=interval, progress=False, auto_adjust=False)
    if data is None or data.empty:
        tkr = yf.Ticker(ticker)
        data = tkr.history(period="max", interval=interval, auto_adjust=False)
    if data is None or data.empty:
        raise ValueError(f"No data returned from Yahoo for '{ticker}'. "
                         f"Tip: check the symbol (e.g., NVDA, or SG tickers like C38U.SI).")

    series = data["Adj Close"] if (adjusted and "Adj Close" in data) else data.get("Close")
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    df = pd.DataFrame({
        "Date": data.index.tz_localize(None),
        "Close": pd.to_numeric(series, errors="coerce")
    }).dropna()
    df = df[df["Close"] > 0].reset_index(drop=True)
    return df

# ---------- Core utils ----------
def compute_bands(series: pd.Series, median_window: int | None = None):
    s = series.dropna()
    if s.empty:
        raise ValueError("No data points available to compute bands.")
    if median_window is None:
        ref = float(s.median())
    else:
        if median_window <= 1:
            raise ValueError("median_window must be > 1")
        roll = s.rolling(window=median_window, min_periods=max(2, median_window // 3)).median()
        last_valid = roll.dropna().iloc[-1] if not roll.dropna().empty else np.nan
        ref = float(last_valid) if np.isfinite(last_valid) else float(s.median())
    bands = {
        "median": ref,
        "+25%": ref * 1.25,
        "-25%": ref * 0.75,
        "+50%": ref * 1.50,
        "-50%": ref * 0.50,
    }
    return ref, bands

def compute_bands_with_history(series: pd.Series):
    s = series.dropna()
    if s.empty:
        raise ValueError("No data for band calculation.")
    median_val = float(s.median())
    hi = float(s.max())
    lo = float(s.min())
    bands = {
        "median": median_val,
        "+25%": median_val * 1.25,
        "-25%": median_val * 0.75,
        "+50%": median_val * 1.50,
        "-50%": median_val * 0.50,
        "high": hi,
        "low": lo,
    }
    return median_val, bands

def _parse_duration_to_days(s: str | None) -> int | None:
    if not s:
        return None
    s = str(s).strip().lower()
    import re as _re
    m = _re.match(r'^(\d+)([dmy]?)$', s)
    if not m:
        raise ValueError("channel-window must look like '180d', '24m', '10y', or a plain number of days.")
    n = int(m.group(1))
    unit = m.group(2) or 'd'
    if unit == 'd':
        return n
    if unit == 'm':
        return int(round(n * 30.4375))
    if unit == 'y':
        return int(round(n * 365.25))
    raise ValueError("Bad unit for channel-window. Use d/m/y.")

def _channel_offsets(dts: pd.Series, prices: pd.Series, low_pct: float, high_pct: float, window_days: int | None):
    dts = pd.to_datetime(dts)
    prices = prices.astype(float)
    if window_days:
        end = dts.iloc[-1]
        start = end - pd.Timedelta(days=window_days)
        msk = dts >= start
        dts = dts[msk]
        prices = prices[msk]
    a, b = _fit_log_line(dts, prices)
    t_days = (dts - dts.iloc[0]).dt.total_seconds().values / 86400.0
    y = np.log(prices.values)
    y_fit = a + b * t_days
    resid = y - y_fit
    low = np.nanpercentile(resid, low_pct)
    high = np.nanpercentile(resid, high_pct)
    return a, b, dts.iloc[0], low, high

def _eval_channel_lines(a: float, b: float, dates_for_eval: pd.Series, t0_date: pd.Timestamp, low: float, high: float, fractions=(0.25, 0.75)):
    dts = pd.to_datetime(dates_for_eval)
    t_days = (dts - t0_date).dt.total_seconds().values / 86400.0
    base = a + b * t_days
    mid_offset = (low + high) / 2.0
    lines = {
        "lower": np.exp(base + low),
        "upper": np.exp(base + high),
        "middle": np.exp(base + mid_offset),
    }
    for f in fractions:
        off = low + f * (high - low)
        lines[f"inner_{int(f*100)}"] = np.exp(base + off)
    return lines

# --- TradingView helpers ---
MYX_CODE_TO_TV = {
    "1295": "PBBANK",  # Public Bank
    "1155": "MAYBANK",
    "1023": "CIMB",
    "5347": "TENAGA",
    "5183": "PETDAG",
    "1961": "TOPGLOV",
}

def tradingview_symbol_from_yahoo(ticker: str) -> str | None:
    t = (ticker or "").upper().strip()
    if t.endswith(".SI"):
        return f"SGX-{t[:-3]}"
    if t.endswith(".KL"):
        code = t[:-3]
        name = MYX_CODE_TO_TV.get(code)
        return f"MYX-{name}" if name else f"MYX-{code}"
    if t in {"^STI", "STI", "^FTSESTI"}:
        return "INDEX-STI"
    if t in {"BTC-USD", "BTCUSD"}:
        return "CRYPTO-BTCUSD"
    if t.isalpha():
        return f"NASDAQ-{t}"
    return None

def tradingview_url(ticker: str) -> str | None:
    sym = tradingview_symbol_from_yahoo(ticker)
    return f"https://www.tradingview.com/symbols/{sym}/" if sym else None

def google_symbol_from_yahoo(tkr: str) -> str:
    t = (tkr or "").upper().strip()
    if t.endswith(".SI"):
        return f"SGX:{t[:-3]}"
    if t.endswith(".KL"):
        return f"KLSE:{t[:-3]}"
    if t.isalpha():
        return f"NASDAQ:{t}"
    return t

def investor_links_for(tkr: str, name_guess: str):
    t = (tkr or "").upper().strip()
    links = []
    tv = tradingview_url(tkr)
    if tv:
        links.append(("TradingView • Chart", tv))
    links.append(("Google • Chart", f"https://www.google.com/finance/quote/{google_symbol_from_yahoo(tkr)}"))
    links.append(("Yahoo • Chart", f"https://finance.yahoo.com/quote/{t}/chart"))

    if t == "J69U.SI":
        links += [
            ("FCT • Financial Information", "https://fct.frasersproperty.com/financial_information.html"),
            ("FCT • Publications", "https://fct.frasersproperty.com/publications.html"),
            ("FCT • Presentations", "https://fct.frasersproperty.com/presentations.html"),
        ]
    if t == "C38U.SI":
        links += [
            ("CICT • Financials", "https://www.cict.com.sg/en/investor-relations/financial-information.html"),
            ("CICT • Reports", "https://www.cict.com.sg/en/investor-relations/financial-information/reports.html"),
        ]
    if t == "M44U.SI":
        links += [
            ("MLT • Financials", "https://www.mapletreelogisticstrust.com/investor-relations/financial-information/"),
            ("MLT • Results & Presentations", "https://www.mapletreelogisticstrust.com/investor-relations/financial-information/results-and-presentations/"),
        ]
    if t == "A17U.SI":
        links += [("AREIT • Financials", "https://www.ascendas-reit.com/en/investor-relations/financial-information/")]

    links += [
        ("Yahoo • Financials", f"https://finance.yahoo.com/quote/{t}/financials"),
        ("Yahoo • Key Statistics", f"https://finance.yahoo.com/quote/{t}/key-statistics"),
        ("Yahoo • Holders", f"https://finance.yahoo.com/quote/{t}/holders"),
    ]
    return links

# --- Main page ticker input (always visible) ---
main_ticker = st.text_input(
    "Yahoo Ticker",
    value=st.session_state.get("last_ticker", "J69U.SI"),
    help="Examples: NVDA (US), J69U.SI (Frasers Centrepoint Trust), M44U.SI (Mapletree Logistics)"
)

with st.sidebar:
    # ---------- State & Callbacks ----------
    if "source" not in st.session_state:
        st.session_state.source = "manual"  # "manual" | "watch"
    if "manual_ticker" not in st.session_state:
        st.session_state.manual_ticker = st.session_state.get("last_ticker", "J69U.SI")
    if "watch_pick" not in st.session_state:
        st.session_state.watch_pick = ""

    def _on_pick_change():
        st.session_state.source = "watch"
        if st.session_state.watch_pick:
            st.session_state.last_ticker = st.session_state.watch_pick
            st.session_state.manual_ticker = st.session_state.watch_pick

    def _on_manual_change():
        st.session_state.source = "manual"
        st.session_state.watch_pick = ""
        st.session_state.last_ticker = st.session_state.manual_ticker

    # ---------- Watchlist select ----------
    st.subheader("📌 Watchlist")
    _watch_keys = list(WATCHLIST.keys())
    _watch_options = [""] + _watch_keys

    st.selectbox(
        "Choose from watchlist",
        options=_watch_options,
        index=_watch_options.index(st.session_state.get("watch_pick", "")),
        key="watch_pick",
        format_func=lambda t: "(None)" if t == "" else f"{WATCHLIST[t]} ({t})",
        on_change=_on_pick_change,
    )

    # ---------- Manual input (always visible) ----------
    st.text_input(
        "Yahoo Ticker (manual entry)",
        key="manual_ticker",
        value=st.session_state.get("manual_ticker", "C38U.SI"),
        help="Examples: NVDA (US), C38U.SI (CICT), M44U.SI (Mapletree Logistics)",
        on_change=_on_manual_change,
    )

    # ---------- Decide final ticker ----------
    ticker = (st.session_state.manual_ticker if st.session_state.source == "manual"
            else st.session_state.watch_pick)
    ticker = (ticker or "").strip().upper()
    display_name = WATCHLIST.get(ticker) or resolve_name(ticker)
    st.session_state["last_ticker"] = ticker

    st.divider()
    st.header("Inputs")
    start = st.text_input("Start Date (YYYY-MM-DD)", value="")
    end = st.text_input("End Date (YYYY-MM-DD)", value="")
    interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
    adjusted = st.checkbox("Use Adjusted Close", value=True)
    mode = st.selectbox("Mode", ["Median Bands", "Historical Bands", "Parallel Channel (log)"], index=2)
    log_scale = st.checkbox("Log scale (y-axis)", value=True)

    cagr_window_label = st.selectbox(
        "CAGR window",
        ["Full period", "1y", "3y", "5y", "10y"]
    )
    _window_map = {"1y": 365, "3y": 365*3, "5y": 365*5, "10y": 365*10}
    cagr_days = None if cagr_window_label == "Full period" else _window_map[cagr_window_label]

    # NEW: CAGR mode toggle
    cagr_mode = st.radio(
        "CAGR mode",
        ["Start–End CAGR", "Trend (Regression) CAGR"],
        index=1,
        help="Start–End uses first & last price only. Trend uses regression of ln(price) over time (all points)."
    )

    # NEW: Lock channel window to CAGR window
    lock_channel_to_cagr = st.checkbox(
        "Lock channel window to CAGR window",
        value=True,
        help="When on, the Parallel Channel uses the same lookback as the selected CAGR window."
    )

    st.caption("Median options")
    median_window = st.number_input("Rolling median window (days, optional)", min_value=0, value=0, step=1)

    st.caption("Channel options")
    low_pct = st.slider("Channel low percentile", min_value=0.0, max_value=50.0, value=0.0, step=0.5)
    high_pct = st.slider("Channel high percentile", min_value=50.0, max_value=100.0, value=100.0, step=0.5)
    channel_window = st.text_input("Channel window (e.g., 36m / 10y / 900d)", value="")
    shade = st.checkbox("Shade middle region", value=True)

    # ===== Daily section overlays =====
    st.caption("Daily chart overlays")
    overlays = st.multiselect(
        "Overlays (Daily section)",
        ["Bollinger Bands", "Ichimoku Cloud"],
        default=[]
    )
    st.session_state["overlays"] = overlays  # so the daily section can read it

    st.caption("Moving Averages (per selected timeframe)")
    ma1_on = st.checkbox("Show MA #1", value=True, key="ma1_on")
    ma1_len = st.number_input("MA #1 length (bars)", min_value=1, value=20, step=1, key="ma1_len")

    ma2_on = st.checkbox("Show MA #2", value=True, key="ma2_on")
    ma2_len = st.number_input("MA #2 length (bars)", min_value=1, value=50, step=1, key="ma2_len")

    # BB params
    if "Bollinger Bands" in overlays:
        with st.expander("Bollinger Bands settings", expanded=True):
            bb_period = st.number_input("BB Period", min_value=5, value=20, key="bb_period")
            bb_std    = st.number_input("BB Std Dev (σ)", min_value=0.5, value=2.0, step=0.5, key="bb_std")
            bb_shade  = st.checkbox("Shade BB range", value=True, key="bb_shade")

    # Ichimoku params
    if "Ichimoku Cloud" in overlays:
        with st.expander("Ichimoku settings", expanded=True):
            ich_tenkan = st.number_input("Tenkan (Conversion)", min_value=2, value=9, key="ich_tenkan")
            ich_kijun  = st.number_input("Kijun (Base)",       min_value=2, value=26, key="ich_kijun")
            ich_senB   = st.number_input("Senkou B",           min_value=2, value=52, key="ich_senB")
            ich_disp   = st.number_input("Displacement (forward)", min_value=1, value=26, key="ich_disp")
            ich_lines  = st.multiselect("Show lines", ["Tenkan", "Kijun", "Chikou"],
                                        default=["Tenkan","Kijun","Chikou"], key="ich_lines")
            ich_shade  = st.checkbox("Shade Cloud", value=True, key="ich_shade")

    if st.button("🔄 Clear cache (daily loader)"):
        try:
            _load_ohlc_from_yahoo_v2.clear()
        except Exception:
            pass
        st.cache_data.clear()
        st.rerun()

    # ===== Daily section timeframe & range =====
    st.caption("Daily chart timeframe & range")
    daily_tf_label = st.selectbox(
        "Timeframe (Daily section)",
        ["Daily", "Weekly", "Monthly"],
        index=0,
        key="daily_tf_label",
    )

    daily_lookback = st.selectbox(
        "Lookback window",
        ["Full history", "1y", "2y", "5y"],
        index=0,
        key="daily_lookback",
    )

st.title("📈 Price Bands & Parallel Channel")
err_box = st.empty()

# ------------------ DATA LOAD & METRICS ------------------
try:
    s = start or None
    e = end or None
    st.caption(f"Fetching: **{ticker}**  • Interval: {interval}  • Adjusted: {adjusted}")
    df = load_prices_from_yahoo(ticker, s, e, interval, adjusted)
    df = df.dropna().sort_values("Date")
    if len(df) < 5:
        st.warning("Not enough points to plot.")
        st.stop()

    # CAGR value according to chosen mode
    if cagr_mode == "Trend (Regression) CAGR":
        cagr = compute_trend_cagr(df["Date"], df["Close"], window_days=cagr_days)
    else:
        cagr = compute_cagr(df["Date"], df["Close"], window_days=cagr_days)

    # --- Ticker ribbon ---
    latest_dt = df["Date"].iloc[-1]
    latest_px = float(df["Close"].iloc[-1])
    prev_px = float(df["Close"].iloc[-2]) if len(df) > 1 else latest_px
    change = latest_px - prev_px
    pct_change = (change / prev_px * 100) if prev_px else 0

    arrow = "🔺" if change > 0 else ("🔻" if change < 0 else "⏺")
    cagr_label = "Trend CAGR" if cagr_mode == "Trend (Regression) CAGR" else "CAGR"
    cagr_text = f"📈 {cagr_label} ({cagr_window_label}): {cagr*100:.2f}%/yr" if cagr is not None else f"📈 {cagr_label}: n/a"

    # Detect currency symbol
    def _currency_for(tkr: str) -> str:
        if tkr.endswith(".KL"): return "RM"
        if tkr.endswith(".SI"): return "S$"
        if tkr.upper() == "BTC-USD": return "USD"
        return "$"
    currency_symbol = _currency_for(ticker)

    st.markdown(
        f"""
        <div style='background-color:#f8f9fa;padding:10px;border-radius:5px;
                    font-size:20px;font-weight:bold;text-align:center;'>
            {display_name} <span style="opacity:0.7">({ticker})</span><br/>
            📅 {latest_dt.date().strftime('%Y-%m-%d')}  |  
            💰 {currency_symbol} {latest_px:,.4f}
            <span style='color:{"green" if change>=0 else "red"}'>
                {arrow} {change:+.4f} ({pct_change:+.2f}%)
            </span><br/>
            {cagr_text}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Fundamentals & dividends
    @st.cache_data(show_spinner=False, ttl=3600)
    def fetch_yf_info_and_dividends(ticker: str):
        tkr = yf.Ticker(ticker)
        try:
            info = tkr.info or {}
        except Exception:
            info = {}
        try:
            dividends = tkr.dividends
        except Exception:
            dividends = pd.Series(dtype=float)
        return info, dividends

    def ttm_dividend_yield(div_series: pd.Series, last_price: float) -> float | None:
        if div_series is None or div_series.empty or not last_price or last_price <= 0:
            return None
        idx = pd.to_datetime(div_series.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert(None)
        cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=365)
        mask = idx >= cutoff
        total_12m = float(div_series[mask].sum())
        return (total_12m / last_price) if total_12m > 0 else None

    def _safe_float(d: dict, key: str):
        v = d.get(key, None)
        try:
            return float(v)
        except Exception:
            return None

    info, dividends = fetch_yf_info_and_dividends(ticker)

    cur = _currency_for(ticker)
    pe_t = _safe_float(info, "trailingPE")
    pe_f = _safe_float(info, "forwardPE")
    pb   = _safe_float(info, "priceToBook")
    eps  = _safe_float(info, "trailingEps")
    payout = _safe_float(info, "payoutRatio")
    dy = ttm_dividend_yield(dividends, latest_px)

    inv_links = investor_links_for(ticker, display_name)

except Exception as ex:
    err_box.error(f"Data error: {ex}")
    st.stop()

# ------------------ MAIN PLOT ------------------
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["Date"], y=df["Close"], mode="lines",
    name="Close", hovertemplate="Date=%{x|%Y-%m-%d}<br>Close=%{y:.4f}<extra></extra>"
))

title_suffix = ""

if mode in ("Median Bands", "Historical Bands"):
    win = None if (not median_window or median_window <= 1) else int(median_window)
    if mode == "Median Bands":
        ref, bands = compute_bands(df["Close"], win)
    else:
        ref, bands = compute_bands_with_history(df["Close"])

    for key, dash in [("median", "dash"), ("+25%", "dot"), ("-25%", "dot"), ("+50%", "dashdot"), ("-50%", "dashdot")]:
        fig.add_hline(y=bands[key], line_dash=dash, annotation_text=key, annotation_position="right")

    if mode == "Historical Bands":
        fig.add_hline(y=bands["high"], line_color="red", line_width=2, annotation_text="Historical High", annotation_position="right")
        fig.add_hline(y=bands["low"], line_color="green", line_width=2, annotation_text="Historical Low", annotation_position="right")

    if shade:
        y0, y1 = bands["-25%"], bands["+25%"]
        fig.add_traces([
            go.Scatter(
                x=list(df["Date"]) + list(df["Date"][::-1]),
                y=[y0]*len(df) + [y1]*len(df),
                fill="toself", opacity=0.08, name="±25% zone",
                hoverinfo="skip", line=dict(width=0)
            )
        ])

elif mode == "Parallel Channel (log)":
    # Decide channel lookback
    if lock_channel_to_cagr:
        wdays = cagr_days  # tie channel to CAGR window
    else:
        wdays = _parse_duration_to_days(channel_window) if channel_window else None

    dffit = df.copy()
    if wdays:
        cutoff = df["Date"].max() - pd.Timedelta(days=wdays)
        dffit = df[df["Date"] >= cutoff].copy()
        if len(dffit) < 5:
            st.warning("Not enough data in channel window; using full series.")
            dffit = df.copy()

    # Clamp percentiles
    low_pct = max(0.0, min(100.0, float(low_pct)))
    high_pct = max(0.0, min(100.0, float(high_pct)))
    if high_pct <= low_pct:
        err_box.error("Channel high percentile must be > low percentile.")
        st.stop()

    a, b, t0_date, low, high = _channel_offsets(dffit["Date"], dffit["Close"], low_pct, high_pct, wdays)
    lines = _eval_channel_lines(a, b, df["Date"], t0_date, low, high, fractions=(0.25, 0.75))
    upper = lines["upper"]; lower = lines["lower"]; mid = lines["middle"]
    q25 = lines["inner_25"]; q75 = lines["inner_75"]

    def add_line(y, name, dash=None, width=2):
        fig.add_trace(go.Scatter(x=df["Date"], y=y, mode="lines", name=name,
                                 line=dict(dash=dash, width=width),
                                 hovertemplate="Date=%{x|%Y-%m-%d}<br>"+name+"=%{y:.4f}<extra></extra>"))
    add_line(upper, f"Upper ({high_pct:.1f}th)", width=3)
    add_line(q75, "75%", dash="dash")
    add_line(mid, "Middle", dash="dot", width=2)
    add_line(q25, "25%", dash="dash")
    add_line(lower, f"Lower ({low_pct:.1f}th)", width=3)

    if shade:
        fig.add_traces([
            go.Scatter(
                x=list(df["Date"]) + list(df["Date"][::-1]),
                y=list(q25) + list(q75[::-1]),
                fill="toself", opacity=0.06, name="25–75 zone",
                hoverinfo="skip", line=dict(width=0)
            ),
            go.Scatter(
                x=list(df["Date"]) + list(df["Date"][::-1]),
                y=list(lower) + list(upper[::-1]),
                fill="toself", opacity=0.03, name="Lower–Upper zone",
                hoverinfo="skip", line=dict(width=0)
            )
        ])

    # Title suffix for clarity
    if lock_channel_to_cagr:
        title_suffix = f" • Channel window: {cagr_window_label}"
    elif channel_window:
        title_suffix = f" • Channel window: {channel_window}"

# Corner annotation (no plotted CAGR line)
if cagr is not None:
    cagr_label = "Trend CAGR" if cagr_mode == "Trend (Regression) CAGR" else "CAGR"
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.99, y=0.99,
        xanchor="right", yanchor="top",
        text=f"{cagr_label} ({cagr_window_label}): {cagr*100:.2f}%/yr",
        showarrow=False,
        font=dict(size=12)
    )

# Layout
fig.update_layout(
    height=600,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=40, r=40, t=60, b=40),
    title=f"{display_name} ({ticker}) • {mode}{title_suffix}",
    xaxis_title="Date",
    yaxis_title="Price",
)
fig.update_yaxes(type="log" if log_scale else "linear")

st.plotly_chart(fig, width="stretch")

st.divider()
st.subheader("Company Snapshot")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Price", f"{cur} {latest_px:,.2f}")
c2.metric("P/E (TTM)", "–" if pe_t is None else f"{pe_t:.2f}")
c3.metric("P/E (Fwd)", "–" if pe_f is None else f"{pe_f:.2f}")
c4.metric("P/B", "–" if pb is None else f"{pb:.2f}")
c5.metric("EPS (TTM)", "–" if eps is None else f"{cur} {eps:.2f}")
c6.metric("Div. Yield (TTM)", "–" if dy is None else f"{dy*100:.2f}%")

if payout is not None and payout > 0:
    st.caption(f"Payout ratio (TTM): {payout*100:.1f}%")

# Recent dividends (last 5)
if dividends is not None and not dividends.empty:
    df_div = dividends.reset_index()
    df_div.columns = ["Date", "Dividend"]
    df_div = df_div.sort_values("Date", ascending=False).head(5)
    st.markdown("**Recent Dividends**")
    st.dataframe(df_div, width="stretch", hide_index=True)
else:
    st.info("No dividend records available from Yahoo for this ticker.")

# Investor Links
st.markdown("**Investor Links**")
if inv_links:
    MAX_PER_ROW = 6
    for start in range(0, len(inv_links), MAX_PER_ROW):
        row = inv_links[start:start + MAX_PER_ROW]
        cols = st.columns(len(row))
        for col, (label, url) in zip(cols, row):
            with col:
                st.markdown(
                    f"<a href='{url}' target='_blank' rel='noopener noreferrer' "
                    f"style='text-decoration:none;font-weight:600;background:#4CAF50;"
                    f"color:white;padding:6px 12px;border-radius:6px;display:inline-block;'>"
                    f"{label}</a>",
                    unsafe_allow_html=True,
                )
else:
    st.info("No links available for this ticker.")

# Small metrics summary
latest_dt = df["Date"].iloc[-1]
latest_px = float(df["Close"].iloc[-1])
st.caption(f"Latest: {latest_px:.4f} on {latest_dt.date()}  •  Points: {len(df)}")
if cagr is not None:
    cagr_label = "Trend CAGR" if cagr_mode == "Trend (Regression) CAGR" else "CAGR"
    st.caption(f"**{cagr_label}:** {cagr*100:.2f}% per year")

# =========================
# 📉 Daily Candlestick + Optional Overlays (BB / Ichimoku)
# =========================
st.divider()
st.subheader("📉 Daily Candles with Overlays")

@st.cache_data(ttl=1800, show_spinner=False)
def _load_ohlc_from_yahoo(ticker: str, start: str | None, end: str | None, interval: str = "1d") -> pd.DataFrame:
    data = yf.download(
        ticker,
        period=None if start else "max",
        start=None if start is None else start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    if data is None or data.empty:
        data = yf.Ticker(ticker).history(period="max", interval=interval, auto_adjust=False)
    if data is None or data.empty:
        return pd.DataFrame()
    try:
        data.index = pd.to_datetime(data.index).tz_localize(None)
    except Exception:
        pass
    if "Close" not in data.columns and "Adj Close" in data.columns:
        data["Close"] = data["Adj Close"]
    keep = [c for c in ["Open","High","Low","Close","Volume"] if c in data.columns]
    out = data[keep].dropna(how="any").copy()
    for c in ["Open","High","Low","Close"]:
        if c in out.columns:
            out[c] = pd.to_numeric(cast := out[c], errors="coerce")
    return out[out["Close"] > 0]

def _resample_ohlc_local(daily_df: pd.DataFrame, tf_label: str) -> pd.DataFrame:
    if daily_df.empty or tf_label == "Daily":
        return daily_df.copy()
    rule = "W-FRI" if tf_label == "Weekly" else "M"
    agg = {}
    for col in ["Open", "High", "Low", "Close"]:
        if col in daily_df.columns:
            agg[col] = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}[col]
    if "Volume" in daily_df.columns:
        agg["Volume"] = "sum"
    res = daily_df.resample(rule).apply(agg).dropna(how="any")
    return res

ohlc_raw = _load_ohlc_from_yahoo_v2(ticker, s, e, interval)

if ohlc_raw.empty:
    st.info("No OHLC data returned for the daily section.")
else:
    ohlc = _resample_ohlc_local(ohlc_raw, st.session_state.get("daily_tf_label", "Daily"))

    lookback_days = {
        "1y": 365,
        "2y": 365 * 2,
        "5y": int(365.25 * 5),
    }
    lb = st.session_state.get("daily_lookback", "Full history")
    if lb in lookback_days and len(ohlc) > 0:
        cutoff = ohlc.index.max() - pd.Timedelta(days=lookback_days[lb])
        ohlc = ohlc[ohlc.index >= cutoff]

    if ohlc.empty:
        st.info("No data for the selected timeframe/range.")
    else:
        fig_daily = go.Figure()

        if {"Open","High","Low","Close"}.issubset(ohlc.columns):
            fig_daily.add_trace(go.Candlestick(
                x=ohlc.index,
                open=ohlc["Open"], high=ohlc["High"], low=ohlc["Low"], close=ohlc["Close"],
                name=f"{st.session_state.get('daily_tf_label','Daily')} Price"
            ))
        else:
            fig_daily.add_trace(go.Scatter(x=ohlc.index, y=ohlc["Close"], name="Close", mode="lines"))

        close_series = ohlc["Close"].astype("float64")

        # Moving Averages
        if "Close" in ohlc.columns:
            if st.session_state.get("ma1_on", False):
                L1 = int(st.session_state.get("ma1_len", 20))
                if L1 > 0 and len(close_series) >= L1:
                    ma1_series = close_series.rolling(L1).mean()
                    mask1 = ma1_series.notna()
                    fig_daily.add_trace(go.Scatter(
                        x=ohlc.index[mask1], y=ma1_series[mask1],
                        name=f"MA{L1}", mode="lines"
                    ))

            if st.session_state.get("ma2_on", False):
                L2 = int(st.session_state.get("ma2_len", 50))
                if L2 > 0 and len(close_series) >= L2:
                    ma2_series = close_series.rolling(L2).mean()
                    mask2 = ma2_series.notna()
                    fig_daily.add_trace(go.Scatter(
                        x=ohlc.index[mask2], y=ma2_series[mask2],
                        name=f"MA{L2}", mode="lines"
                    ))

        # Bollinger Bands
        overlays_local = st.session_state.get("overlays", [])
        if "Bollinger Bands" in overlays_local and len(close_series) >= int(st.session_state.get("bb_period", 20)):
            _bbp = int(st.session_state.get("bb_period", 20))
            _bbs = float(st.session_state.get("bb_std", 2.0))
            ma = close_series.rolling(_bbp).mean()
            sd = close_series.rolling(_bbp).std(ddof=0)
            up = ma + _bbs * sd
            dn = ma - _bbs * sd
            mask = ma.notna() & up.notna() & dn.notna()
            xvals = ohlc.index[mask]

            fig_daily.add_trace(go.Scatter(x=xvals, y=ma[mask], name=f"BB MA ({_bbp})", mode="lines"))
            fig_daily.add_trace(go.Scatter(x=xvals, y=dn[mask], name=f"BB Lower ({_bbs}σ)", mode="lines",
                                           line=dict(width=1), opacity=0.5))
            if st.session_state.get("bb_shade", True):
                fig_daily.add_trace(go.Scatter(
                    x=xvals, y=up[mask], name=f"BB Upper ({_bbs}σ)", mode="lines",
                    line=dict(width=1), opacity=0.5, fill='tonexty', fillcolor='rgba(200,200,200,0.15)'
                ))
            else:
                fig_daily.add_trace(go.Scatter(x=xvals, y=up[mask], name=f"BB Upper ({_bbs}σ)", mode="lines",
                                               line=dict(width=1), opacity=0.5))

        # Ichimoku Cloud
        if "Ichimoku Cloud" in overlays_local and {"High","Low","Close"}.issubset(ohlc.columns):
            _tenkan = int(st.session_state.get("ich_tenkan", 9))
            _kijun  = int(st.session_state.get("ich_kijun", 26))
            _senB   = int(st.session_state.get("ich_senB", 52))
            _disp   = int(st.session_state.get("ich_disp", 26))
            _lines  = set(st.session_state.get("ich_lines", ["Tenkan","Kijun","Chikou"]))
            _shade  = bool(st.session_state.get("ich_shade", True))

            tenkan_sen = (ohlc["High"].rolling(_tenkan).max() + ohlc["Low"].rolling(_tenkan).min()) / 2.0
            kijun_sen  = (ohlc["High"].rolling(_kijun).max()  + ohlc["Low"].rolling(_kijun).min())  / 2.0
            sen_a = ((tenkan_sen + kijun_sen) / 2.0).shift(_disp)
            sen_b = ((ohlc["High"].rolling(_senB).max() + ohlc["Low"].rolling(_senB).min()) / 2.0).shift(_disp)
            chikou = ohlc["Close"].shift(-_disp)

            if "Tenkan" in _lines:
                fig_daily.add_trace(go.Scatter(x=ohlc.index, y=tenkan_sen, name=f"Tenkan ({_tenkan})", mode="lines"))
            if "Kijun" in _lines:
                fig_daily.add_trace(go.Scatter(x=ohlc.index, y=kijun_sen, name=f"Kijun ({_kijun})", mode="lines"))
            if "Chikou" in _lines:
                fig_daily.add_trace(go.Scatter(x=ohlc.index, y=chikou, name=f"Chikou ({_disp} back)", mode="lines", opacity=0.6))

            mask = sen_a.notna() & sen_b.notna()
            if mask.any():
                xvals = ohlc.index[mask]
                lower = np.minimum(sen_a[mask], sen_b[mask])
                upper = np.maximum(sen_a[mask], sen_b[mask])
                fig_daily.add_trace(go.Scatter(x=xvals, y=lower, name="Senkou Lower", mode="lines",
                                               line=dict(width=1), opacity=0.4))
                if _shade:
                    fig_daily.add_trace(go.Scatter(x=xvals, y=upper, name="Senkou Upper", mode="lines",
                                                   line=dict(width=1), opacity=0.4,
                                                   fill='tonexty', fillcolor='rgba(160,200,255,0.18)'))
                else:
                    fig_daily.add_trace(go.Scatter(x=xvals, y=upper, name="Senkou Upper", mode="lines",
                                                   line=dict(width=1), opacity=0.4))

        fig_daily.update_layout(
            height=650,
            title=f"{display_name} ({ticker}) • {st.session_state.get('daily_tf_label','Daily')} "
                  f"({st.session_state.get('daily_lookback','Full history')})",
            xaxis_title="Date",
            yaxis_title="Price",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=40, r=40, t=60, b=40),
        )
        fig_daily.update_yaxes(type="log" if log_scale else "linear")
        st.plotly_chart(fig_daily, use_container_width=True)