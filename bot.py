import time
import os
import json
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
NY_TZ = ZoneInfo("America/New_York")
STATE_FILE = "triggered_setups.json"

PORTFOLIO = {
    "XAU/USD": {
        "symbol": "XAU/USD",
        "pip": 0.1,
        "digits": 2,
        "name": "GOLD (Spot)"
    }
}

TIMEFRAME = "15min"
OUTPUTSIZE = 400                     # 400 candle 15m = 100 jam
MAX_RUNTIME_SECONDS = 330 * 60       # 5.5 jam (19.800 detik)
ACTIVE_CHECK_INTERVAL = 300          # Sinkron tiap 5 menit saat sesi aktif
OFFHOURS_CHECK_INTERVAL = 900        # Sinkron tiap 15 menit saat Sydney/Off-hours
MAX_FVG_ALERT_AGE_BARS = 3           # Maksimal usia FVG (3 bar = 45 menit)

HTTP_SESSION = requests.Session()


# ----------------- UTILITAS KEAMANAN, STATE & TELEGRAM -----------------
def sanitize_log(text: str) -> str:
    safe_text = str(text)
    for secret in (TWELVE_API_KEY, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID):
        if secret:
            safe_text = safe_text.replace(secret, "***REDACTED***")
    return safe_text


def load_bot_state() -> tuple[set, str]:
    """Memuat daftar setup yang sudah dikirim dan status pasar terakhir (OPEN/CLOSED)."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Kompatibel dengan format list lama maupun dict baru
                if isinstance(data, list):
                    return set(data[-200:]), "UNKNOWN"
                if isinstance(data, dict):
                    setups = set(data.get("triggered_setups", [])[-200:])
                    market_status = data.get("market_status", "UNKNOWN")
                    return setups, market_status
        except Exception as e:
            print(f"Gagal memuat state file: {sanitize_log(e)}")
    return set(), "UNKNOWN"


def save_bot_state(triggered: set, market_status: str):
    """Menyimpan daftar setup dan status pasar (OPEN/CLOSED) ke file JSON."""
    try:
        payload = {
            "market_status": market_status,
            "triggered_setups": list(triggered)[-200:]
        }
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        print(f"Gagal menyimpan state file: {sanitize_log(e)}")


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


# ----------------- MANAJEMEN WAKTU & SESI MARKET (NY DST-SAFE) -----------------
def is_forex_market_closed(now_utc: datetime = None) -> bool:
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    now_ny = now_utc.astimezone(NY_TZ)
    weekday = now_ny.weekday()
    hour = now_ny.hour

    # Pasar tutup Jumat 17:00 waktu NY hingga Minggu 17:00 waktu NY
    if weekday == 5:
        return True
    if weekday == 4 and hour >= 17:
        return True
    if weekday == 6 and hour < 17:
        return True
    return False


def get_current_session(now_utc: datetime = None) -> str:
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    now_ny = now_utc.astimezone(NY_TZ)
    ny_hour = now_ny.hour

    sessions = []
    if 19 <= ny_hour or ny_hour < 4:
        sessions.append("Tokyo 🇯🇵")
    if 2 <= ny_hour < 11:
        sessions.append("London 🇬🇧")
    if 8 <= ny_hour < 17:
        sessions.append("New York 🇺🇸")
    return " + ".join(sessions) if sessions else "Sydney/Off-hours 🇦🇺"


def is_in_ict_killzone(now_utc: datetime = None) -> tuple[bool, str]:
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    now_ny = now_utc.astimezone(NY_TZ)
    ny_hour = now_ny.hour

    if 2 <= ny_hour < 5:
        return True, "London Killzone 🔥"
    if 7 <= ny_hour < 11:
        return True, "NY Killzone 🔥"
    return False, "Di Luar Killzone"


def get_clock_aligned_sleep(interval_seconds: int, max_remaining: float) -> float:
    now_ts = time.time()
    next_slot = ((int(now_ts) // interval_seconds) + 1) * interval_seconds + 5
    sleep_sec = max(5.0, next_slot - now_ts)
    return min(sleep_sec, max_remaining)


# ----------------- DATA FETCHING & INDIKATOR -----------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = (100 - (100 / (1 + rs))).fillna(50)

    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean().bfill()
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
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
        if len(df) < 100:
            print(f"Data {symbol} terlalu sedikit ({len(df)} baris).")
            return None

        return add_indicators(df)
    except Exception as e:
        print(f"Gagal mengambil {symbol}: {sanitize_log(e)}")
        return None


def resample_to_1h(df_15m: pd.DataFrame) -> pd.DataFrame:
    df_htf = (
        df_15m.set_index("datetime")
        .resample("1h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    )
    if len(df_htf) > 2 and df_15m.iloc[0]["datetime"].minute != 0:
        df_htf = df_htf.iloc[1:]

    df_htf = df_htf.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return add_indicators(df_htf)


# ----------------- ALGORITMA STRUKTUR SMC & ICT -----------------
def find_fractals(df: pd.DataFrame, left: int = 2, right: int = 2):
    swing_highs = []
    swing_lows = []
    highs = df["high"].values
    lows = df["low"].values

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


def extract_structural_range(df_struct: pd.DataFrame, atr_ref: float):
    sh_list, sl_list = find_fractals(df_struct, left=2, right=2)
    sh_val = sh_list[-1][1] if sh_list else None
    sl_val = sl_list[-1][1] if sl_list else None

    if sh_val is None or sl_val is None or sh_val <= sl_val or (sh_val - sl_val) < (0.8 * atr_ref):
        window = df_struct.iloc[-45:]
        sh_val = float(window["high"].max())
        sl_val = float(window["low"].min())

    if sh_val <= sl_val:
        return None, None
    return sh_val, sl_val


def analyze_htf_1h(df_15m: pd.DataFrame):
    df_1h = resample_to_1h(df_15m)
    closed_1h = df_1h.iloc[:-1].copy().reset_index(drop=True)

    if len(closed_1h) < 30:
        return {"bias": "NEUTRAL", "zone": "EQ", "model": "Wait", "sh": None, "sl": None, "eq": None}

    struct_1h = closed_1h.iloc[:-3].reset_index(drop=True)
    action_1h = closed_1h.iloc[-3:]
    last_1h = closed_1h.iloc[-1]
    atr_1h = float(last_1h["atr"])

    htf_sh, htf_sl = extract_structural_range(struct_1h, atr_1h)
    if htf_sh is None or htf_sl is None:
        return {"bias": "NEUTRAL", "zone": "EQ", "model": "Wait", "sh": None, "sl": None, "eq": None}

    htf_eq = (htf_sh + htf_sl) / 2.0
    live_price = float(df_15m.iloc[-1]["close"])
    zone = "DISCOUNT" if live_price <= htf_eq else "PREMIUM"

    htf_swept_ssl = (action_1h["low"] < htf_sl).any() and (last_1h["close"] > htf_sl)
    htf_swept_bsl = (action_1h["high"] > htf_sh).any() and (last_1h["close"] < htf_sh)

    htf_bull_bos = (action_1h["close"] > htf_sh).any() and (last_1h["close"] > htf_eq)
    htf_bear_bos = (action_1h["close"] < htf_sl).any() and (last_1h["close"] < htf_eq)

    if htf_swept_ssl and not htf_swept_bsl:
        bias, model_1h = "BULLISH", "SSL Sweep"
    elif htf_swept_bsl and not htf_swept_ssl:
        bias, model_1h = "BEARISH", "BSL Sweep"
    elif htf_bull_bos:
        bias, model_1h = "BULLISH", "1H BOS"
    elif htf_bear_bos:
        bias, model_1h = "BEARISH", "1H BOS"
    elif last_1h["close"] > last_1h["ema_50"]:
        bias, model_1h = "BULLISH", "EMA50 Trend"
    elif last_1h["close"] < last_1h["ema_50"]:
        bias, model_1h = "BEARISH", "EMA50 Trend"
    else:
        bias, model_1h = "NEUTRAL", "Sideways"

    active_sh = max(htf_sh, float(action_1h["high"].max()))
    active_sl = min(htf_sl, float(action_1h["low"].min()))

    return {
        "bias": bias,
        "model": model_1h,
        "zone": zone,
        "sh": active_sh,
        "sl": active_sl,
        "eq": (active_sh + active_sl) / 2.0
    }


def generate_trade_setup(pair: str, df: pd.DataFrame, cfg: dict):
    digits = cfg["digits"]
    pip_size = cfg["pip"]
    live_candle = df.iloc[-1]
    live_close = float(live_candle["close"])
    live_low = float(live_candle["low"])
    live_high = float(live_candle["high"])

    htf = analyze_htf_1h(df)
    htf_bias = htf["bias"]
    htf_zone = htf["zone"]
    htf_summary = f"{htf_bias} ({htf_zone} • {htf['model']})"

    closed_df = df.iloc[:-1].copy().reset_index(drop=True)
    last_closed = closed_df.iloc[-1]
    atr = float(last_closed["atr"])
    ema50 = float(last_closed["ema_50"])

    struct_15m = closed_df.iloc[-68:-8].reset_index(drop=True)
    action_15m = closed_df.iloc[-8:]

    swing_high, swing_low = extract_structural_range(struct_15m, atr)
    if swing_high is None or swing_low is None:
        return {
            "has_setup": False,
            "order": "WAIT / NO SETUP ⚪",
            "range_15m": "-",
            "setup_id": None,
            "is_fresh": False,
            "htf_info": htf_summary,
            "note": "Menunggu pembentukan struktur 15M."
        }

    range_str = f"{swing_low:.{digits}f} - {swing_high:.{digits}f}"

    bull_origin_idx = int(action_15m["low"].idxmin())
    bear_origin_idx = int(action_15m["high"].idxmax())

    sweep_low_val = float(closed_df.loc[bull_origin_idx, "low"])
    sweep_high_val = float(closed_df.loc[bear_origin_idx, "high"])

    origin_bull_candle = closed_df.loc[bull_origin_idx]
    origin_bear_candle = closed_df.loc[bear_origin_idx]

    after_bull_origin = closed_df.loc[bull_origin_idx + 1:]
    after_bear_origin = closed_df.loc[bear_origin_idx + 1:]

    # PERBAIKAN 1: Konfirmasi MSS menggunakan Struktur Fractal Mikro sebelum Origin Sweep
    pre_bull_df = closed_df.loc[max(0, bull_origin_idx - 10):bull_origin_idx].reset_index(drop=True)
    micro_sh_list, _ = find_fractals(pre_bull_df, left=1, right=1)
    mss_bull_level = micro_sh_list[-1][1] if micro_sh_list else float(
        closed_df.loc[max(0, bull_origin_idx - 2):bull_origin_idx, "high"].max()
    )

    pre_bear_df = closed_df.loc[max(0, bear_origin_idx - 10):bear_origin_idx].reset_index(drop=True)
    _, micro_sl_list = find_fractals(pre_bear_df, left=1, right=1)
    mss_bear_level = micro_sl_list[-1][1] if micro_sl_list else float(
        closed_df.loc[max(0, bear_origin_idx - 2):bear_origin_idx, "low"].min()
    )

    swept_ssl_15m = (sweep_low_val < swing_low) and (last_closed["close"] > swing_low)
    mss_bull_reversal = swept_ssl_15m and len(after_bull_origin) > 0 and (
        after_bull_origin["close"] > mss_bull_level
    ).any()
    bos_bull_cont = (action_15m["close"] > swing_high).any() and (last_closed["close"] > ema50)
    bullish_valid = mss_bull_reversal or bos_bull_cont

    swept_bsl_15m = (sweep_high_val > swing_high) and (last_closed["close"] < swing_high)
    mss_bear_reversal = swept_bsl_15m and len(after_bear_origin) > 0 and (
        after_bear_origin["close"] < mss_bear_level
    ).any()
    bos_bear_cont = (action_15m["close"] < swing_low).any() and (last_closed["close"] < ema50)
    bearish_valid = mss_bear_reversal or bos_bear_cont

    # PERBAIKAN 2: Equilibrium (EQ) dihitung spesifik dari Leg Impulsif Aktif
    bull_leg_high = float(after_bull_origin["high"].max()) if len(after_bull_origin) > 0 else swing_high
    bull_eq_15m = (sweep_low_val + max(bull_leg_high, swing_high)) / 2.0

    bear_leg_low = float(after_bear_origin["low"].min()) if len(after_bear_origin) > 0 else swing_low
    bear_eq_15m = (sweep_high_val + min(bear_leg_low, swing_low)) / 2.0

    bull_fvg = None
    bear_fvg = None

    start_scan = max(0, len(closed_df) - 9)
    end_scan = len(closed_df) - 2

    for i in range(start_scan, end_scan):
        c1 = closed_df.iloc[i]
        c2 = closed_df.iloc[i + 1]
        c3 = closed_df.iloc[i + 2]

        c2_body = abs(c2["close"] - c2["open"])
        is_displacement = c2_body > (0.40 * atr)
        after_fvg = closed_df.iloc[i + 3:]
        fvg_age_bars = (len(closed_df) - 1) - (i + 2)

        # Seleksi Extreme Bullish FVG
        if (
            bull_fvg is None
            and (i + 1) >= bull_origin_idx
            and is_displacement
            and (c2["close"] > c2["open"])
            and (c3["low"] - c1["high"]) > (0.12 * atr)
        ):
            fvg_bot = float(c1["high"])
            fvg_top = float(c3["low"])
            ce_bull = (fvg_bot + fvg_top) / 2.0
            structural_low = min(sweep_low_val, float(c1["low"]), float(c2["low"]))
            planned_entry = fvg_top if fvg_top <= bull_eq_15m else ce_bull

            unviolated = True
            already_used = False

            if len(after_fvg) > 0:
                if (after_fvg["close"] < fvg_bot).any() or (after_fvg["low"] <= structural_low).any():
                    unviolated = False
                for idx_after, row_after in after_fvg.iterrows():
                    if row_after["low"] <= planned_entry:
                        # PERBAIKAN 3: Evaluasi pantulan hanya pada candle SETELAH candle penjemput FVG
                        subsequent = closed_df.loc[idx_after + 1:]
                        if len(subsequent) > 0 and (subsequent["high"] >= planned_entry + (0.8 * atr)).any():
                            already_used = True
                            break

            if unviolated and not already_used and (live_low > structural_low):
                origin_id = origin_bull_candle["datetime"].strftime("%Y%m%d_%H%M")
                is_fresh = fvg_age_bars <= MAX_FVG_ALERT_AGE_BARS
                bull_fvg = (fvg_bot, fvg_top, ce_bull, structural_low, origin_id, is_fresh)

        # Seleksi Extreme Bearish FVG
        if (
            bear_fvg is None
            and (i + 1) >= bear_origin_idx
            and is_displacement
            and (c2["close"] < c2["open"])
            and (c1["low"] - c3["high"]) > (0.12 * atr)
        ):
            fvg_bot = float(c3["high"])
            fvg_top = float(c1["low"])
            ce_bear = (fvg_bot + fvg_top) / 2.0
            structural_high = max(sweep_high_val, float(c1["high"]), float(c2["high"]))
            planned_entry = fvg_bot if fvg_bot >= bear_eq_15m else ce_bear

            unviolated = True
            already_used = False

            if len(after_fvg) > 0:
                if (after_fvg["close"] > fvg_top).any() or (after_fvg["high"] >= structural_high).any():
                    unviolated = False
                for idx_after, row_after in after_fvg.iterrows():
                    if row_after["high"] >= planned_entry:
                        # PERBAIKAN 3: Evaluasi pantulan hanya pada candle SETELAH candle penjemput FVG
                        subsequent = closed_df.loc[idx_after + 1:]
                        if len(subsequent) > 0 and (subsequent["low"] <= planned_entry - (0.8 * atr)).any():
                            already_used = True
                            break

            if unviolated and not already_used and (live_high < structural_high):
                origin_id = origin_bear_candle["datetime"].strftime("%Y%m%d_%H%M")
                is_fresh = fvg_age_bars <= MAX_FVG_ALERT_AGE_BARS
                bear_fvg = (fvg_bot, fvg_top, ce_bear, structural_high, origin_id, is_fresh)

    in_kz, kz_name = is_in_ict_killzone()
    default_setup = {
        "has_setup": False,
        "order": "WAIT / NO SETUP ⚪",
        "range_15m": range_str,
        "setup_id": None,
        "is_fresh": False,
        "htf_info": htf_summary,
        "note": "Menunggu sweep likuiditas atau displacement FVG baru."
    }

    # Evaluasi Setup Bullish XAU/USD
    if htf_bias == "BULLISH" and bullish_valid and bull_fvg:
        fvg_bot, fvg_top, fvg_ce, st_low, origin_id, is_fresh = bull_fvg
        entry_price = fvg_top if fvg_top <= bull_eq_15m else fvg_ce
        sl_price = st_low - (0.20 * atr)
        risk = entry_price - sl_price

        if risk > 0 and live_close > sl_price:
            # PERBAIKAN 4: Proyeksi Dinamis TP1 saat skenario BOS jika swing_high terlalu dekat
            if (swing_high - entry_price) >= (1.3 * risk):
                tp1_price = swing_high
            elif bos_bull_cont:
                tp1_price = entry_price + (1.6 * risk)
            else:
                default_setup["note"] = "Setup Bullish diabaikan (R:R ke BSL < 1:1.3)."
                return default_setup

            rr_tp1 = (tp1_price - entry_price) / risk
            htf_dol = htf["sh"] if (htf["sh"] and htf["sh"] >= entry_price + (2.5 * risk)) else (entry_price + (3.0 * risk))
            tp2_price = max(tp1_price + risk, htf_dol)
            rr_tp2 = (tp2_price - entry_price) / risk

            if live_high >= tp1_price and live_low > fvg_top:
                default_setup["note"] = "Target TP1 tersentuh sebelum mitigasi FVG (Front-run)."
                return default_setup

            sl_pips = int(round(risk / pip_size))
            tp1_pips = int(round((tp1_price - entry_price) / pip_size))
            tp2_pips = int(round((tp2_price - entry_price) / pip_size))

            is_aplus = (htf_zone == "DISCOUNT" or swept_ssl_15m) and in_kz
            grade = f"A+ {kz_name}" if is_aplus else "B-Grade Setup ⚠️"
            model_tag = f"SSL Sweep @ {st_low:.{digits}f} + MSS" if mss_bull_reversal else f"15M BOS + FVG (SL @ {st_low:.{digits}f})"

            return {
                "has_setup": True,
                "direction": "BUY LIMIT 🟢",
                "grade": grade,
                "order": f"BUY LIMIT 🟢 [{grade}]",
                "entry_val": f"{entry_price:.{digits}f}",
                "fvg_zone": f"{fvg_bot:.{digits}f} - {fvg_top:.{digits}f}",
                "sl": f"{sl_price:.{digits}f} (-{sl_pips} Pips)",
                "tp1": f"{tp1_price:.{digits}f} (+{tp1_pips} Pips • 1:{rr_tp1:.1f}R)",
                "tp2": f"{tp2_price:.{digits}f} (+{tp2_pips} Pips • 1:{rr_tp2:.1f}R)",
                "range_15m": range_str,
                "setup_id": f"{pair}_BULL_{origin_id}",
                "is_fresh": is_fresh,
                "htf_info": htf_summary,
                "note": model_tag
            }

    # Evaluasi Setup Bearish XAU/USD
    elif htf_bias == "BEARISH" and bearish_valid and bear_fvg:
        fvg_bot, fvg_top, fvg_ce, st_high, origin_id, is_fresh = bear_fvg
        entry_price = fvg_bot if fvg_bot >= bear_eq_15m else fvg_ce
        sl_price = st_high + (0.20 * atr)
        risk = sl_price - entry_price

        if risk > 0 and live_close < sl_price:
            # PERBAIKAN 4: Proyeksi Dinamis TP1 saat skenario BOS jika swing_low terlalu dekat
            if (entry_price - swing_low) >= (1.3 * risk):
                tp1_price = swing_low
            elif bos_bear_cont:
                tp1_price = entry_price - (1.6 * risk)
            else:
                default_setup["note"] = "Setup Bearish diabaikan (R:R ke SSL < 1:1.3)."
                return default_setup

            rr_tp1 = (entry_price - tp1_price) / risk
            htf_dol = htf["sl"] if (htf["sl"] and htf["sl"] <= entry_price - (2.5 * risk)) else (entry_price - (3.0 * risk))
            tp2_price = min(tp1_price - risk, htf_dol)
            rr_tp2 = (entry_price - tp2_price) / risk

            if live_low <= tp1_price and live_high < fvg_bot:
                default_setup["note"] = "Target TP1 tersentuh sebelum mitigasi FVG (Front-run)."
                return default_setup

            sl_pips = int(round(risk / pip_size))
            tp1_pips = int(round((entry_price - tp1_price) / pip_size))
            tp2_pips = int(round((entry_price - tp2_price) / pip_size))

            is_aplus = (htf_zone == "PREMIUM" or swept_bsl_15m) and in_kz
            grade = f"A+ {kz_name}" if is_aplus else "B-Grade Setup ⚠️"
            model_tag = f"BSL Sweep @ {st_high:.{digits}f} + MSS" if mss_bear_reversal else f"15M BOS + FVG (SH @ {st_high:.{digits}f})"

            return {
                "has_setup": True,
                "direction": "SELL LIMIT 🔴",
                "grade": grade,
                "order": f"SELL LIMIT 🔴 [{grade}]",
                "entry_val": f"{entry_price:.{digits}f}",
                "fvg_zone": f"{fvg_bot:.{digits}f} - {fvg_top:.{digits}f}",
                "sl": f"{sl_price:.{digits}f} (-{sl_pips} Pips)",
                "tp1": f"{tp1_price:.{digits}f} (+{tp1_pips} Pips • 1:{rr_tp1:.1f}R)",
                "tp2": f"{tp2_price:.{digits}f} (+{tp2_pips} Pips • 1:{rr_tp2:.1f}R)",
                "range_15m": range_str,
                "setup_id": f"{pair}_BEAR_{origin_id}",
                "is_fresh": is_fresh,
                "htf_info": htf_summary,
                "note": model_tag
            }

    return default_setup


# ----------------- FORMATTER PESAN RINGKAS & JELAS -----------------
def build_single_shot_alert(pair: str, setup: dict, now_utc: datetime) -> str:
    now_wib = now_utc.astimezone(WIB_TZ)
    return (
        f"🚨 <b>{html.escape(setup['direction'])} {pair} — [{html.escape(setup['grade'])}]</b>\n"
        f"🕒 <code>{now_wib.strftime('%H:%M')} WIB</code> | Bias 1H: <b>{html.escape(setup['htf_info'])}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 <b>Entry Limit :</b> <code>{html.escape(setup['entry_val'])}</code>\n"
        f"📦 <b>Zona FVG    :</b> <code>{html.escape(setup['fvg_zone'])}</code>\n"
        f"🛑 <b>Stop Loss   :</b> <code>{html.escape(setup['sl'])}</code>\n"
        f"💰 <b>TP 1 (BEP)  :</b> <code>{html.escape(setup['tp1'])}</code>\n"
        f"🚀 <b>TP 2 (DOL)  :</b> <code>{html.escape(setup['tp2'])}</code>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💡 <b>Konfluensi:</b> <i>{html.escape(setup['note'])}</i>\n"
        f"⚠️ <i>Risiko maks 0.5%-1% ekuitas. Batal jika TP1 tersentuh sebelum harga masuk zona Entry.</i>"
    )


def build_actionable_dashboard(data_dict: dict, setups_cache: dict, header_title: str = None):
    valid_pairs = {k: v for k, v in data_dict.items() if v is not None}
    if not valid_pairs:
        return None

    now_utc = datetime.now(timezone.utc)
    now_wib = now_utc.astimezone(WIB_TZ)
    session = get_current_session(now_utc)
    _, kz_label = is_in_ict_killzone(now_utc)

    title = header_title or "⚡ <b>OUTLOOK XAU/USD (UPDATE 3 JAM)</b>"
    msg = (
        f"{title}\n"
        f"🕒 <code>{now_wib.strftime('%H:%M')} WIB</code> | Sesi: <b>{session}</b> ({kz_label})\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
    )

    for pair, df in valid_pairs.items():
        cfg = PORTFOLIO[pair]
        curr = df.iloc[-1]
        setup = setups_cache.get(pair) or generate_trade_setup(pair, df, cfg)
        digits = cfg["digits"]

        msg += f"🏆 <b>{pair}</b> — Live: <code>{curr['close']:.{digits}f}</code> (ATR: <code>{curr['atr']:.1f}</code>)\n"
        msg += f"• <b>Bias 1H     :</b> <code>{html.escape(setup['htf_info'])}</code>\n"
        msg += f"• <b>Rentang 15M :</b> <code>{html.escape(setup['range_15m'])}</code>\n"
        msg += f"• <b>Status      :</b> <b>{html.escape(setup['order'])}</b>\n"

        if setup.get("has_setup"):
            msg += f"• <b>Entry Limit :</b> <code>{html.escape(setup['entry_val'])}</code> (FVG: <code>{html.escape(setup['fvg_zone'])}</code>)\n"
            msg += f"• <b>SL / TP1    :</b> <code>{html.escape(setup['sl'])}</code> / <code>{html.escape(setup['tp1'])}</code>\n"

        msg += f"• <b>Catatan     :</b> <i>{html.escape(str(setup['note']))}</i>\n"

    return msg


# ----------------- ENGINE LOOP UTAMA (5.5 HOURS RELAY) -----------------
def main():
    start_time = time.time()
    triggered_setups, market_status = load_bot_state()

    init_utc = datetime.now(timezone.utc)
    init_block = (init_utc.date(), init_utc.hour // 3)

    is_manual_start = (GITHUB_EVENT_NAME != "repository_dispatch")
    if not is_manual_start and not (init_utc.hour % 3 == 0 and init_utc.minute < 16):
        last_reported_block = init_block
    else:
        last_reported_block = None

    first_cycle = True

    print(
        f"XAU/USD Engine dimulai pada {init_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC "
        f"(Event: {GITHUB_EVENT_NAME}, Market Status Tersimpan: {market_status})."
    )

    while (time.time() - start_time) < MAX_RUNTIME_SECONDS:
        try:
            now_utc = datetime.now(timezone.utc)

            # 1. PENANGANAN SAAT PASAR TUTUP (HANYA KIRIM 1X DI AWAL TUTUP)
            if is_forex_market_closed(now_utc):
                if market_status != "CLOSED":
                    print("Pasar baru saja tutup. Mengambil snapshot penutupan XAU/USD (1x)...")
                    data_cache = {}
                    setups_cache = {}
                    for pair, cfg in PORTFOLIO.items():
                        df = fetch_twelve_data(cfg["symbol"])
                        data_cache[pair] = df
                        if df is not None:
                            setups_cache[pair] = generate_trade_setup(pair, df, cfg)

                    dashboard = build_actionable_dashboard(
                        data_cache,
                        setups_cache,
                        header_title="⏸️ <b>PASAR XAU/USD TUTUP (AKHIR PEKAN)</b>\n<i>Snapshot penutupan pasar. Bot masuk mode tidur hingga Senin pagi:</i>"
                    )
                    if dashboard:
                        send_telegram(dashboard)

                    market_status = "CLOSED"
                    save_bot_state(triggered_setups, market_status)

                first_cycle = False
                remaining = MAX_RUNTIME_SECONDS - (time.time() - start_time)
                if remaining <= 0:
                    break
                time.sleep(min(OFFHOURS_CHECK_INTERVAL, remaining))
                continue

            # 2. PENANGANAN SAAT PASAR AKTIF & DETEKSI MARKET BARU BUKA
            just_reopened = (market_status == "CLOSED")
            if market_status != "OPEN":
                market_status = "OPEN"
                save_bot_state(triggered_setups, market_status)

            data_cache = {}
            setups_cache = {}

            for pair, cfg in PORTFOLIO.items():
                df = fetch_twelve_data(cfg["symbol"])
                data_cache[pair] = df

                if df is not None:
                    setup = generate_trade_setup(pair, df, cfg)
                    setups_cache[pair] = setup

                    setup_id = setup.get("setup_id")
                    is_fresh = setup.get("is_fresh", False)

                    if setup_id and is_fresh and setup_id not in triggered_setups:
                        alert_msg = build_single_shot_alert(pair, setup, now_utc)
                        send_telegram(alert_msg)
                        triggered_setups.add(setup_id)
                        save_bot_state(triggered_setups, market_status)
                        print(f"Sinyal 1x terkirim dan dikunci: {setup_id}")

            # 3. KIRIM PESAN MARKET BUKA (1X) ATAU LAPORAN RUTIN 3 JAM
            current_block = (now_utc.date(), now_utc.hour // 3)
            if just_reopened:
                open_msg = build_actionable_dashboard(
                    data_cache,
                    setups_cache,
                    header_title="🔔 <b>PASAR XAU/USD KEMBALI BUKA (MARKET OPEN)</b>\n<i>Pemindaian otomatis sesi awal pekan telah aktif kembali:</i>"
                )
                if open_msg:
                    send_telegram(open_msg)
                    last_reported_block = current_block
                    print("Notifikasi Market Open berhasil dikirim.")
            elif current_block != last_reported_block:
                is_startup_msg = first_cycle and is_manual_start
                hdr = "🟢 <b>XAU/USD ENGINE ACTIVE — INITIAL SCAN</b>" if is_startup_msg else None
                dashboard = build_actionable_dashboard(data_cache, setups_cache, header_title=hdr)
                if dashboard:
                    send_telegram(dashboard)
                    last_reported_block = current_block
                    print(f"Dashboard rutin terkirim (Blok={current_block}).")

            first_cycle = False

        except Exception as e:
            print(f"Error pada loop utama: {sanitize_log(e)}")

        remaining_time = MAX_RUNTIME_SECONDS - (time.time() - start_time)
        if remaining_time <= 0:
            break

        current_session = get_current_session()
        interval = OFFHOURS_CHECK_INTERVAL if "Sydney" in current_session else ACTIVE_CHECK_INTERVAL
        actual_sleep = get_clock_aligned_sleep(interval, remaining_time)

        if actual_sleep > 0:
            time.sleep(actual_sleep)

    print("Siklus 5.5 jam selesai dengan normal (exit 0). Memicu Auto-Relay ke runner berikutnya...")


if __name__ == "__main__":
    main()
