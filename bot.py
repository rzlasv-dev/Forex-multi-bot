import time
import os
import html
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# ----------------- KONFIGURASI UTAMA -----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY")
GITHUB_EVENT_NAME = os.getenv("GITHUB_EVENT_NAME", "manual")

WIB_TZ = ZoneInfo("Asia/Jakarta")

PORTFOLIO = {
    "XAU/USD": {"symbol": "XAU/USD", "pip": 0.1, "digits": 2, "name": "GOLD (Spot)"},
    "EUR/USD": {"symbol": "EUR/USD", "pip": 0.0001, "digits": 5, "name": "EUR/USD"},
    "USD/JPY": {"symbol": "USD/JPY", "pip": 0.01, "digits": 3, "name": "USD/JPY"}
}

TIMEFRAME = "15min"
OUTPUTSIZE = 400                     # 400 candle 15m = 100 jam (cukup untuk 1H tanpa ekstra API call)
MAX_RUNTIME_SECONDS = 330 * 60       # 5.5 jam (19.800 detik)
ACTIVE_CHECK_INTERVAL = 300          # Cek tiap 5 menit saat Tokyo/London/NY
OFFHOURS_CHECK_INTERVAL = 900        # Cek tiap 15 menit saat Sydney/Off-hours

HTTP_SESSION = requests.Session()


# ----------------- UTILITAS KEAMANAN & TELEGRAM -----------------
def sanitize_log(text: str) -> str:
    safe_text = str(text)
    for secret in (TWELVE_API_KEY, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID):
        if secret:
            safe_text = safe_text.replace(secret, "***REDACTED***")
    return safe_text


def send_telegram(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Peringatan: TELEGRAM_TOKEN atau TELEGRAM_CHAT_ID belum diset di Secrets!")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message[:4090],
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    try:
        res = HTTP_SESSION.post(url, json=payload, timeout=12)
        if res.status_code != 200:
            print(f"Gagal kirim Telegram (HTTP {res.status_code}): {sanitize_log(res.text)}")
    except Exception as e:
        print(f"Error koneksi Telegram: {sanitize_log(e)}")


# ----------------- MANAJEMEN WAKTU & SESI MARKET -----------------
def is_forex_market_closed(now_utc: datetime = None) -> bool:
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    weekday = now_utc.weekday()
    hour = now_utc.hour
    if weekday == 5:
        return True
    if weekday == 4 and hour >= 22:
        return True
    if weekday == 6 and hour < 22:
        return True
    return False


def get_current_session(now_utc: datetime = None) -> str:
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    hour = now_utc.hour
    sessions = []
    if 0 <= hour < 9:
        sessions.append("Tokyo 🇯🇵")
    if 7 <= hour < 16:
        sessions.append("London 🇬🇧")
    if 13 <= hour < 22:
        sessions.append("New York 🇺🇸")
    return " + ".join(sessions) if sessions else "Sydney/Off-hours 🇦🇺"


def is_in_ict_killzone(now_utc: datetime = None) -> bool:
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    hour = now_utc.hour
    return (7 <= hour <= 10) or (12 <= hour <= 16)


# ----------------- DATA FETCHING & INDIKATOR -----------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

    delta = df['close'].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = (100 - (100 / (1 + rs))).fillna(50)

    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/14, adjust=False).mean().bfill()
    return df


def fetch_twelve_data(symbol: str):
    if not TWELVE_API_KEY:
        print("Error: TWELVE_API_KEY belum diset di GitHub Secrets!")
        return None

    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": TIMEFRAME,
        "outputsize": OUTPUTSIZE,
        "timezone": "UTC",
        "apikey": TWELVE_API_KEY
    }
    try:
        resp = HTTP_SESSION.get(url, params=params, timeout=15)
        res = resp.json()
        if "values" not in res:
            err_msg = res.get("message", "Unknown API response")
            print(f"Error Twelve Data ({symbol}): {sanitize_log(err_msg)}")
            return None

        df = pd.DataFrame(res["values"]).iloc[::-1].reset_index(drop=True)
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=['open', 'high', 'low', 'close']).reset_index(drop=True)
        if len(df) < 80:
            print(f"Data {symbol} terlalu sedikit ({len(df)} baris).")
            return None

        return add_indicators(df)
    except Exception as e:
        print(f"Gagal mengambil {symbol}: {sanitize_log(e)}")
        return None


def resample_to_1h(df_15m: pd.DataFrame) -> pd.DataFrame:
    """Merakit DataFrame 1H (HTF) utuh kompatibel dengan Pandas 2.0+"""
    df_htf = (
        df_15m.set_index('datetime')
        .resample('1h')
        .agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})
    )
    if len(df_htf) > 2 and df_15m.iloc[0]['datetime'].minute != 0:
        df_htf = df_htf.iloc[1:]

    df_htf = df_htf.dropna(subset=['open', 'high', 'low', 'close']).reset_index()
    return add_indicators(df_htf)


# ----------------- ALGORITMA SMC & ICT -----------------
def find_fractals(df: pd.DataFrame, left: int = 2, right: int = 2):
    swing_highs = []
    swing_lows = []
    highs = df['high'].values
    lows = df['low'].values

    for i in range(left, len(df) - right):
        is_high = all(highs[i] > highs[i - j] for j in range(1, left + 1)) and \
                  all(highs[i] >= highs[i + j] for j in range(1, right + 1))
        is_low = all(lows[i] < lows[i - j] for j in range(1, left + 1)) and \
                 all(lows[i] <= lows[i + j] for j in range(1, right + 1))
        if is_high:
            swing_highs.append((i, float(highs[i])))
        if is_low:
            swing_lows.append((i, float(lows[i])))
    return swing_highs, swing_lows


def extract_valid_dealing_range(df_closed: pd.DataFrame, sh_list: list, sl_list: list):
    if not sh_list or not sl_list:
        return None, None, None, None

    sh_idx, sh_val = sh_list[-1]
    sl_idx, sl_val = sl_list[-1]
    atr = df_closed.iloc[-1]['atr']

    if sh_val <= sl_val or (sh_val - sl_val) < (0.8 * atr):
        window = df_closed.iloc[-40:]
        sh_idx = int(window['high'].idxmax())
        sl_idx = int(window['low'].idxmin())
        sh_val = float(df_closed.loc[sh_idx, 'high'])
        sl_val = float(df_closed.loc[sl_idx, 'low'])

    if sh_val <= sl_val:
        return None, None, None, None

    return sh_idx, sh_val, sl_idx, sl_val


def analyze_htf_1h(df_15m: pd.DataFrame):
    df_1h = resample_to_1h(df_15m)
    closed_1h = df_1h.iloc[:-1].copy().reset_index(drop=True)

    if len(closed_1h) < 25:
        return {"bias": "NEUTRAL", "zone": "EQ", "sh": None, "sl": None, "eq": None}

    sh_list, sl_list = find_fractals(closed_1h, left=2, right=2)
    _, htf_sh, _, htf_sl = extract_valid_dealing_range(closed_1h, sh_list, sl_list)

    if htf_sh is None or htf_sl is None:
        return {"bias": "NEUTRAL", "zone": "EQ", "sh": None, "sl": None, "eq": None}

    htf_eq = (htf_sh + htf_sl) / 2.0
    last_1h = closed_1h.iloc[-1]
    live_price = float(df_15m.iloc[-1]['close'])
    recent_1h = closed_1h.iloc[-4:]

    zone = "DISCOUNT" if live_price <= htf_eq else "PREMIUM"

    htf_swept_ssl = (recent_1h['low'] < htf_sl).any() and (last_1h['close'] > htf_sl)
    htf_swept_bsl = (recent_1h['high'] > htf_sh).any() and (last_1h['close'] < htf_sh)

    htf_bull_bos = last_1h['close'] > htf_sh
    htf_bear_bos = last_1h['close'] < htf_sl

    if htf_swept_ssl and not htf_swept_bsl:
        bias = "BULLISH"
    elif htf_swept_bsl and not htf_swept_ssl:
        bias = "BEARISH"
    elif htf_bull_bos:
        bias = "BULLISH"
    elif htf_bear_bos:
        bias = "BEARISH"
    elif last_1h['close'] > last_1h['ema_50']:
        bias = "BULLISH"
    elif last_1h['close'] < last_1h['ema_50']:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"

    return {
        "bias": bias,
        "zone": zone,
        "sh": htf_sh,
        "sl": htf_sl,
        "eq": htf_eq
    }


def generate_trade_setup(pair: str, df: pd.DataFrame, cfg: dict):
    digits = cfg["digits"]
    live_candle = df.iloc[-1]
    live_close = float(live_candle['close'])
    live_low = float(live_candle['low'])
    live_high = float(live_candle['high'])

    htf = analyze_htf_1h(df)
    htf_bias = htf["bias"]
    htf_zone = htf["zone"]
    htf_summary = f"1H: {htf_bias} ({htf_zone})"

    closed_df = df.iloc[:-1].copy().reset_index(drop=True)
    last_closed = closed_df.iloc[-1]
    atr = float(last_closed['atr'])
    ema50 = float(last_closed['ema_50'])

    lookback_df = closed_df.iloc[-60:].reset_index(drop=True)
    sh_15m, sl_15m = find_fractals(lookback_df, left=2, right=2)
    _, swing_high, _, swing_low = extract_valid_dealing_range(lookback_df, sh_15m, sl_15m)

    if swing_high is None or swing_low is None:
        return {
            "order": "WAIT / NO SETUP ⚪",
            "entry": f"{live_close:.{digits}f}",
            "sl": "-", "tp1": "-", "tp2": "-",
            "setup_id": None,
            "stage": "NONE",
            "htf_info": htf_summary,
            "note": "Menunggu pembentukan struktur Dealing Range 15M."
        }

    eq_15m = (swing_high + swing_low) / 2.0

    recent_8 = closed_df.iloc[-8:]
    swept_ssl_mask = recent_8['low'] < swing_low
    swept_bsl_mask = recent_8['high'] > swing_high

    swept_ssl_15m = swept_ssl_mask.any() and (last_closed['close'] > swing_low)
    swept_bsl_15m = swept_bsl_mask.any() and (last_closed['close'] < swing_high)

    bos_bull_mask = closed_df.iloc[-6:]['close'] > swing_high
    bos_bear_mask = closed_df.iloc[-6:]['close'] < swing_low

    bullish_mss = (bos_bull_mask.any() or swept_ssl_15m) and (last_closed['close'] > ema50)
    bearish_mss = (bos_bear_mask.any() or swept_bsl_15m) and (last_closed['close'] < ema50)

    bull_origin_idx = int(recent_8['low'].idxmin())
    bear_origin_idx = int(recent_8['high'].idxmax())

    bull_fvg = None
    bear_fvg = None

    start_scan = max(0, len(closed_df) - 9)
    end_scan = len(closed_df) - 2

    for i in range(start_scan, end_scan):
        c1 = closed_df.iloc[i]
        c2 = closed_df.iloc[i + 1]
        c3 = closed_df.iloc[i + 2]

        c2_body = abs(c2['close'] - c2['open'])
        is_displacement = c2_body > (0.40 * atr)
        after_closed = closed_df.iloc[i + 3:]

        # Bullish FVG
        if (i + 1) >= bull_origin_idx and is_displacement and (c3['low'] - c1['high']) > (0.12 * atr):
            fvg_bot = float(c1['high'])
            fvg_top = float(c3['low'])
            ce_bull = (fvg_bot + fvg_top) / 2.0
            ob_low = float(min(c1['low'], c2['low']))
            planned_entry = fvg_top if fvg_top <= eq_15m else ce_bull

            unviolated = True
            already_used = False

            if len(after_closed) > 0:
                if (after_closed['close'] < fvg_bot).any() or (after_closed['low'] <= ob_low).any():
                    unviolated = False
                older_closed = after_closed.iloc[:-1]
                if len(older_closed) > 0 and (older_closed['low'] <= planned_entry).any():
                    if (after_closed['high'] >= planned_entry + atr).any():
                        already_used = True

            if unviolated and not already_used and (live_low > ob_low):
                fvg_time = c2['datetime'].strftime('%Y%m%d_%H%M')
                bull_fvg = (fvg_bot, fvg_top, ce_bull, ob_low, fvg_time)

        # Bearish FVG
        if (i + 1) >= bear_origin_idx and is_displacement and (c1['low'] - c3['high']) > (0.12 * atr):
            fvg_bot = float(c3['high'])
            fvg_top = float(c1['low'])
            ce_bear = (fvg_bot + fvg_top) / 2.0
            ob_high = float(max(c1['high'], c2['high']))
            planned_entry = fvg_bot if fvg_bot >= eq_15m else ce_bear

            unviolated = True
            already_used = False

            if len(after_closed) > 0:
                if (after_closed['close'] > fvg_top).any() or (after_closed['high'] >= ob_high).any():
                    unviolated = False
                older_closed = after_closed.iloc[:-1]
                if len(older_closed) > 0 and (older_closed['high'] >= planned_entry).any():
                    if (after_closed['low'] <= planned_entry - atr).any():
                        already_used = True

            if unviolated and not already_used and (live_high < ob_high):
                fvg_time = c2['datetime'].strftime('%Y%m%d_%H%M')
                bear_fvg = (fvg_bot, fvg_top, ce_bear, ob_high, fvg_time)

    in_kz = is_in_ict_killzone()
    setup = {
        "order": "WAIT / NO SETUP ⚪",
        "entry": f"{live_close:.{digits}f}",
        "sl": "-", "tp1": "-", "tp2": "-",
        "setup_id": None,
        "stage": "NONE",
        "htf_info": htf_summary,
        "note": f"15M Range: {swing_low:.{digits}f} - {swing_high:.{digits}f}"
    }

    if htf_bias == "BULLISH" and bullish_mss and bull_fvg:
        fvg_bot, fvg_top, fvg_ce, ob_low, fvg_id = bull_fvg
        entry_price = fvg_top if fvg_top <= eq_15m else fvg_ce
        sl_price = ob_low - (0.15 * atr)
        risk = entry_price - sl_price

        if risk > 0:
            tp1_price = max(swing_high, entry_price + (1.5 * risk))
            htf_dol = htf["sh"] if (htf["sh"] and htf["sh"] >= entry_price + (2.5 * risk)) else (entry_price + (3.0 * risk))
            tp2_price = htf_dol

            is_aplus = (htf_zone == "DISCOUNT") and in_kz
            grade = " [A+ ICT 🔥]" if is_aplus else " [B-Grade ⚠️]"

            in_fvg_zone = (
                (fvg_bot - 0.05 * atr <= live_low <= fvg_top + 0.05 * atr)
                or (fvg_bot <= last_closed['low'] <= fvg_top and live_close <= entry_price + 0.3 * atr)
            ) and (live_close > sl_price)

            if in_fvg_zone:
                order_type = f"BUY NOW (In FVG) 🟢{grade}"
                stage = "NOW"
            elif live_close > fvg_top:
                order_type = f"BUY LIMIT (FVG) 🟢{grade}"
                stage = "LIMIT"
            else:
                return setup

            setup = {
                "order": order_type,
                "entry": f"{entry_price:.{digits}f}",
                "sl": f"{sl_price:.{digits}f}",
                "tp1": f"{tp1_price:.{digits}f}",
                "tp2": f"{tp2_price:.{digits}f}",
                "setup_id": f"{pair}_BULL_{fvg_id}_{stage}",
                "stage": stage,
                "htf_info": htf_summary,
                "note": f"1H Bullish + 15M FVG ({fvg_bot:.{digits}f}-{fvg_top:.{digits}f})"
            }

    elif htf_bias == "BEARISH" and bearish_mss and bear_fvg:
        fvg_bot, fvg_top, fvg_ce, ob_high, fvg_id = bear_fvg
        entry_price = fvg_bot if fvg_bot >= eq_15m else fvg_ce
        sl_price = ob_high + (0.15 * atr)
        risk = sl_price - entry_price

        if risk > 0:
            tp1_price = min(swing_low, entry_price - (1.5 * risk))
            htf_dol = htf["sl"] if (htf["sl"] and htf["sl"] <= entry_price - (2.5 * risk)) else (entry_price - (3.0 * risk))
            tp2_price = htf_dol

            is_aplus = (htf_zone == "PREMIUM") and in_kz
            grade = " [A+ ICT 🔥]" if is_aplus else " [B-Grade ⚠️]"

            in_fvg_zone = (
                (fvg_bot - 0.05 * atr <= live_high <= fvg_top + 0.05 * atr)
                or (fvg_bot <= last_closed['high'] <= fvg_top and live_close >= entry_price - 0.3 * atr)
            ) and (live_close < sl_price)

            if in_fvg_zone:
                order_type = f"SELL NOW (In FVG) 🔴{grade}"
                stage = "NOW"
            elif live_close < fvg_bot:
                order_type = f"SELL LIMIT (FVG) 🔴{grade}"
                stage = "LIMIT"
            else:
                return setup

            setup = {
                "order": order_type,
                "entry": f"{entry_price:.{digits}f}",
                "sl": f"{sl_price:.{digits}f}",
                "tp1": f"{tp1_price:.{digits}f}",
                "tp2": f"{tp2_price:.{digits}f}",
                "setup_id": f"{pair}_BEAR_{fvg_id}_{stage}",
                "stage": stage,
                "htf_info": htf_summary,
                "note": f"1H Bearish + 15M FVG ({fvg_bot:.{digits}f}-{fvg_top:.{digits}f})"
            }

    return setup


# ----------------- FORMATTER DASHBOARD & SINYAL -----------------
def build_actionable_dashboard(data_dict: dict, setups_cache: dict, is_startup: bool = False):
    valid_pairs = {k: v for k, v in data_dict.items() if v is not None}
    if not valid_pairs:
        return None

    now_utc = datetime.now(timezone.utc)
    now_wib = now_utc.astimezone(WIB_TZ)
    session = get_current_session(now_utc)
    kz_status = "ACTIVE 🔥" if is_in_ict_killzone(now_utc) else "INACTIVE 💤"

    if is_startup:
        msg = "🟢 <b>ENGINE ACTIVE ✅ — INITIAL MARKET SCAN</b>\n"
    else:
        msg = "⚡ <b>ICT &amp; SMC 3-HOURLY MARKET OUTLOOK (1H ➔ 15M)</b> ⚡\n"

    msg += f"Sesi: <b>{session}</b> | Killzone: <b>{kz_status}</b>\n"
    msg += f"Waktu: <code>{now_wib.strftime('%H:%M')} WIB</code> (<code>{now_utc.strftime('%H:%M')} UTC</code>)\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━\n\n"

    for pair, df in valid_pairs.items():
        cfg = PORTFOLIO[pair]
        curr = df.iloc[-1]
        setup = setups_cache.get(pair) or generate_trade_setup(pair, df, cfg)
        digits = cfg["digits"]
        safe_note = html.escape(str(setup['note']))

        msg += f"🎯 <b>{pair}</b> | Live Spot: <code>{curr['close']:.{digits}f}</code>\n"
        msg += f"• <b>HTF Bias:</b> <code>{html.escape(setup['htf_info'])}</code>\n"
        msg += f"• <b>Status:</b> <code>{html.escape(setup['order'])}</code>\n"
        msg += f"• <b>Entry:</b> <code>{setup['entry']}</code>\n"
        msg += f"• <b>Stop Loss (SL):</b> <code>{setup['sl']}</code>\n"
        msg += f"• <b>TP1 (Internal):</b> <code>{setup['tp1']}</code>\n"
        msg += f"• <b>TP2 (1H DOL):</b> <code>{setup['tp2']}</code>\n"
        msg += f"• <b>Kondisi:</b> RSI <code>{curr['rsi']:.1f}</code> | <i>{safe_note}</i>\n"
        msg += "──────────────────────\n"

    msg += "<i>Jadwal: Update rutin tiap 3 jam | Sinyal Setup dikirim Instan</i>"
    return msg


# ----------------- ENGINE LOOP UTAMA (5.5 HOURS RELAY) -----------------
def main():
    start_time = time.time()
    triggered_setups = set()
    weekend_notified = False

    init_utc = datetime.now(timezone.utc)
    init_block = (init_utc.date(), init_utc.hour // 3)

    # Jika diaktifkan manual/push (bukan repository_dispatch dari estafet otomatis),
    # kirimkan pesan ENGINE ACTIVE ✅ di siklus pertama!
    is_manual_start = (GITHUB_EVENT_NAME != "repository_dispatch")
    if not is_manual_start and not (init_utc.hour % 3 == 0 and init_utc.minute < 16):
        last_reported_block = init_block
    else:
        last_reported_block = None

    first_cycle = True

    print(
        f"Engine dimulai pada {init_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC "
        f"(Event: {GITHUB_EVENT_NAME}). Batas Relay: {MAX_RUNTIME_SECONDS // 60} menit."
    )

    while (time.time() - start_time) < MAX_RUNTIME_SECONDS:
        try:
            now_utc = datetime.now(timezone.utc)

            # 1. Mode Akhir Pekan (Hemat Kuota API)
            if is_forex_market_closed(now_utc):
                if not weekend_notified:
                    print("Pasar tutup (Akhir Pekan). Mengambil snapshot penutupan...")
                    data_cache = {}
                    setups_cache = {}
                    for pair, cfg in PORTFOLIO.items():
                        df = fetch_twelve_data(cfg["symbol"])
                        data_cache[pair] = df
                        if df is not None:
                            setups_cache[pair] = generate_trade_setup(pair, df, cfg)

                    dashboard = build_actionable_dashboard(data_cache, setups_cache, is_startup=first_cycle)
                    if dashboard:
                        send_telegram(
                            "⏸️ <b>MODE AKHIR PEKAN (PASAR TUTUP)</b>\n"
                            "<i>Berikut posisi penutupan pasar terakhir sebelum mode hemat kuota:</i>\n\n"
                            + dashboard
                        )
                    weekend_notified = True
                    first_cycle = False

                remaining = MAX_RUNTIME_SECONDS - (time.time() - start_time)
                if remaining <= 0:
                    break
                print("Pasar Forex sedang tutup. Tidur 15 menit...")
                time.sleep(min(OFFHOURS_CHECK_INTERVAL, remaining))
                continue

            weekend_notified = False
            data_cache = {}
            setups_cache = {}

            # 2. Pemindaian Market & Deteksi Setup Instan
            for pair, cfg in PORTFOLIO.items():
                df = fetch_twelve_data(cfg["symbol"])
                data_cache[pair] = df

                if df is not None:
                    setup = generate_trade_setup(pair, df, cfg)
                    setups_cache[pair] = setup

                    setup_id = setup.get("setup_id")
                    if setup_id and setup_id not in triggered_setups:
                        now_wib = now_utc.astimezone(WIB_TZ)
                        header = (
                            f"🚨 <b>ICT INSTANT EXECUTION: {pair}</b> 🚨"
                            if setup["stage"] == "NOW"
                            else f"📌 <b>ICT NEW FVG SETUP (LIMIT): {pair}</b> 📌"
                        )
                        alert_msg = (
                            f"{header}\n"
                            f"Waktu: <code>{now_wib.strftime('%H:%M')} WIB</code> (<code>{now_utc.strftime('%H:%M')} UTC</code>)\n\n"
                            f"• <b>HTF Context:</b> <code>{html.escape(setup['htf_info'])}</code>\n"
                            f"• <b>Action:</b> <code>{html.escape(setup['order'])}</code>\n"
                            f"• <b>Entry:</b> <code>{setup['entry']}</code>\n"
                            f"• <b>Stop Loss (SL):</b> <code>{setup['sl']}</code>\n"
                            f"• <b>TP1 (1.5R+):</b> <code>{setup['tp1']}</code>\n"
                            f"• <b>TP2 (1H DOL):</b> <code>{setup['tp2']}</code>\n"
                            f"• <b>Note:</b> <i>{html.escape(setup['note'])}</i>"
                        )
                        send_telegram(alert_msg)
                        triggered_setups.add(setup_id)

            # 3. Laporan Startup Perdana ATAU Laporan Rutin Setiap 3 Jam Sekali
            current_block = (now_utc.date(), now_utc.hour // 3)
            if current_block != last_reported_block:
                is_startup_msg = first_cycle and is_manual_start
                dashboard = build_actionable_dashboard(data_cache, setups_cache, is_startup=is_startup_msg)
                if dashboard:
                    send_telegram(dashboard)
                    last_reported_block = current_block
                    print(f"Dashboard berhasil dikirim (Startup={is_startup_msg}, Blok={current_block}).")

            first_cycle = False

        except Exception as e:
            print(f"Error pada loop utama: {sanitize_log(e)}")

        # 4. Adaptive Sleep dengan Pengaman Waktu Relay
        remaining_time = MAX_RUNTIME_SECONDS - (time.time() - start_time)
        if remaining_time <= 0:
            break

        current_session = get_current_session()
        sleep_duration = OFFHOURS_CHECK_INTERVAL if "Sydney" in current_session else ACTIVE_CHECK_INTERVAL
        actual_sleep = min(sleep_duration, remaining_time)

        if actual_sleep > 0:
            time.sleep(actual_sleep)

    print("Siklus 5.5 jam selesai dengan normal (exit 0). Memicu Auto-Relay ke runner berikutnya...")


if __name__ == "__main__":
    main()
