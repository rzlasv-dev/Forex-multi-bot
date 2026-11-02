import time
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ----------------- KONFIGURASI -----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY")

PORTFOLIO = {
    "XAU/USD": {"symbol": "XAU/USD", "pip": 0.1, "digits": 2, "name": "GOLD (Spot Exness)"},
    "EUR/USD": {"symbol": "EUR/USD", "pip": 0.0001, "digits": 5, "name": "EUR/USD"},
    "USD/JPY": {"symbol": "USD/JPY", "pip": 0.01, "digits": 3, "name": "USD/JPY"}
}

TIMEFRAME = "15min"
MAX_RUNTIME_SECONDS = 330 * 60       # 5.5 jam relay GitHub Actions
ACTIVE_CHECK_INTERVAL = 300          # Cek tiap 5 menit saat sesi aktif
OFFHOURS_CHECK_INTERVAL = 900        # Cek tiap 15 menit saat Sydney/Off-hours (Hemat Kuota API)


def send_telegram(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Peringatan: TELEGRAM_TOKEN atau TELEGRAM_CHAT_ID kosong!")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        res = requests.post(url, json=payload, timeout=10)
        if res.status_code != 200:
            print(f"Gagal kirim Telegram: {res.text}")
    except Exception as e:
        print(f"Error Telegram: {e}")


def is_forex_market_closed():
    """Pasar Forex tutup dari Jumat 22:00 UTC hingga Minggu 22:00 UTC"""
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


def get_current_session():
    hour = datetime.now(timezone.utc).hour
    sessions = []
    if 0 <= hour < 9:
        sessions.append("Tokyo 🇯🇵")
    if 7 <= hour < 16:
        sessions.append("London 🇬🇧")
    if 13 <= hour < 22:
        sessions.append("New York 🇺🇸")
    return " + ".join(sessions) if sessions else "Sydney/Off-hours 🇦🇺"


def is_in_ict_killzone():
    """Validasi ICT Killzone: London Open (07:00-10:00 UTC) & NY Open (12:00-16:00 UTC)"""
    hour = datetime.now(timezone.utc).hour
    return (7 <= hour <= 10) or (12 <= hour <= 16)


def add_indicators(df):
    """Menghitung EMA 50, Wilder's RSI 14, dan Wilder's ATR 14 pada DataFrame OHLC"""
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
    df['atr'] = tr.ewm(alpha=1/14, adjust=False).mean()
    return df


def fetch_twelve_data(symbol):
    """Mengambil 400 candle 15m (setara 100 candle 1H) dengan 1 kredit API"""
    if not TWELVE_API_KEY:
        print("Error: TWELVE_API_KEY belum diset di GitHub Secrets!")
        return None

    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": TIMEFRAME,
        "outputsize": 400,  # 400 candle 15m = 100 jam (Cukup untuk struktur 1H tanpa ekstra API call)
        "apikey": TWELVE_API_KEY
    }
    try:
        res = requests.get(url, params=params, timeout=10).json()
        if "values" not in res:
            print(f"Error Twelve Data ({symbol}): {res.get('message')}")
            return None

        df = pd.DataFrame(res["values"]).iloc[::-1].reset_index(drop=True)
        df['datetime'] = pd.to_datetime(df['datetime'])
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].astype(float)

        return add_indicators(df)
    except Exception as e:
        print(f"Gagal mengambil {symbol}: {e}")
        return None


def resample_to_1h(df_15m):
    """Merakit DataFrame 1H (HTF) langsung dari data 15m (Zero Extra API Call)"""
    df_htf = (
        df_15m.set_index('datetime')
        .resample('1h')
        .agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})
        .dropna()
        .reset_index()
    )
    return add_indicators(df_htf)


def find_fractals(df, left=2, right=2):
    """Mendeteksi Swing High dan Swing Low berbasis 5-bar Fractal"""
    swing_highs = []
    swing_lows = []
    for i in range(left, len(df) - right):
        is_high = all(df.iloc[i]['high'] > df.iloc[i - j]['high'] for j in range(1, left + 1)) and \
                  all(df.iloc[i]['high'] >= df.iloc[i + j]['high'] for j in range(1, right + 1))
        is_low = all(df.iloc[i]['low'] < df.iloc[i - j]['low'] for j in range(1, left + 1)) and \
                 all(df.iloc[i]['low'] <= df.iloc[i + j]['low'] for j in range(1, right + 1))
        if is_high:
            swing_highs.append((i, df.iloc[i]['high']))
        if is_low:
            swing_lows.append((i, df.iloc[i]['low']))
    return swing_highs, swing_lows


def analyze_htf_1h(df_15m):
    """Menganalisa Struktur 1H, Dealing Range (Premium/Discount), dan Liquidity Sweep"""
    df_1h = resample_to_1h(df_15m)
    closed_1h = df_1h.iloc[:-1].copy()

    if len(closed_1h) < 30:
        return {"bias": "NEUTRAL", "zone": "EQ", "sh": None, "sl": None, "eq": None, "poi_active": False}

    sh_list, sl_list = find_fractals(closed_1h, left=2, right=2)
    if not sh_list or not sl_list:
        return {"bias": "NEUTRAL", "zone": "EQ", "sh": None, "sl": None, "eq": None, "poi_active": False}

    _, htf_sh = sh_list[-1]
    _, htf_sl = sl_list[-1]
    htf_eq = (htf_sh + htf_sl) / 2.0

    last_1h = closed_1h.iloc[-1]
    live_price = df_15m.iloc[-1]['close']
    recent_1h = closed_1h.iloc[-4:]

    # Zona Dealing Range 1H
    zone = "DISCOUNT" if live_price <= htf_eq else "PREMIUM"

    # Deteksi 1H BOS atau 1H Liquidity Sweep (Turtle Soup)
    htf_swept_ssl = (recent_1h['low'] < htf_sl).any() and (last_1h['close'] > htf_sl)
    htf_swept_bsl = (recent_1h['high'] > htf_sh).any() and (last_1h['close'] < htf_sh)

    htf_bull_bos = (recent_1h['close'] > htf_sh).any() or (last_1h['close'] > last_1h['ema_50'])
    htf_bear_bos = (recent_1h['close'] < htf_sl).any() or (last_1h['close'] < last_1h['ema_50'])

    bias = "NEUTRAL"
    if htf_swept_ssl or htf_bull_bos:
        bias = "BULLISH"
    elif htf_swept_bsl or htf_bear_bos:
        bias = "BEARISH"

    return {
        "bias": bias,
        "zone": zone,
        "sh": htf_sh,
        "sl": htf_sl,
        "eq": htf_eq,
        "swept_liquidity": htf_swept_ssl or htf_swept_bsl
    }


def generate_trade_setup(pair, df, cfg):
    digits = cfg["digits"]
    live_candle = df.iloc[-1]
    live_close = live_candle['close']
    live_low = live_candle['low']
    live_high = live_candle['high']

    # 1. Analisa Bias Higher Timeframe (1H)
    htf = analyze_htf_1h(df)
    htf_bias = htf["bias"]
    htf_zone = htf["zone"]

    # 2. Evaluasi Struktur Lower Timeframe (15M) pada candle yang sudah close
    closed_df = df.iloc[:-1].copy()
    last_closed = closed_df.iloc[-1]
    atr = last_closed['atr']
    ema50 = last_closed['ema_50']

    sh_15m, sl_15m = find_fractals(closed_df.iloc[-60:], left=2, right=2)
    if not sh_15m or not sl_15m:
        return {
            "order": "WAIT / NO SETUP ⚪",
            "entry": f"{live_close:.{digits}f}",
            "sl": "-", "tp1": "-", "tp2": "-",
            "setup_id": None,
            "htf_info": f"1H {htf_bias} ({htf_zone})",
            "note": "Menunggu pembentukan struktur Fractal 15M."
        }

    _, swing_high = sh_15m[-1]
    _, swing_low = sl_15m[-1]
    eq_15m = (swing_high + swing_low) / 2.0

    # 3. Konfirmasi 15M BOS / Liquidity Sweep
    recent_5 = closed_df.iloc[-5:]
    swept_ssl_15m = (closed_df.iloc[-8:]['low'] < swing_low).any() and (last_closed['close'] > swing_low)
    swept_bsl_15m = (closed_df.iloc[-8:]['high'] > swing_high).any() and (last_closed['close'] < swing_high)

    bullish_mss = ((recent_5['close'] > swing_high).any() or swept_ssl_15m) and (last_closed['close'] > ema50)
    bearish_mss = ((recent_5['close'] < swing_low).any() or swept_bsl_15m) and (last_closed['close'] < ema50)

    # 4. Deteksi Unmitigated FVG 15M dengan Validasi Displacement & Consequent Encroachment (50% CE)
    bull_fvg = None
    bear_fvg = None

    for i in range(len(closed_df) - 8, len(closed_df) - 2):
        c1 = closed_df.iloc[i]
        c2 = closed_df.iloc[i + 1]
        c3 = closed_df.iloc[i + 2]

        c2_body = abs(c2['close'] - c2['open'])
        is_displacement = c2_body > (0.45 * atr)
        after_fvg = df.iloc[i + 3:]  # Mencakup candle live agar FVG yang jebol langsung gugur

        # Bullish FVG
        if is_displacement and (c3['low'] - c1['high']) > (0.15 * atr):
            ce_bull = (c1['high'] + c3['low']) / 2.0
            if (after_fvg['low'] > ce_bull).all():
                fvg_time = c2['datetime'].strftime('%Y%m%d_%H%M')
                bull_fvg = (c1['high'], c3['low'], ce_bull, min(c1['low'], c2['low']), fvg_time)

        # Bearish FVG
        if is_displacement and (c1['low'] - c3['high']) > (0.15 * atr):
            ce_bear = (c3['high'] + c1['low']) / 2.0
            if (after_fvg['high'] < ce_bear).all():
                fvg_time = c2['datetime'].strftime('%Y%m%d_%H%M')
                bear_fvg = (c3['high'], c1['low'], ce_bear, max(c1['high'], c2['high']), fvg_time)

    in_kz = is_in_ict_killzone()
    htf_summary = f"1H: {htf_bias} ({htf_zone})"

    setup = {
        "order": "WAIT / NO SETUP ⚪",
        "entry": f"{live_close:.{digits}f}",
        "sl": "-", "tp1": "-", "tp2": "-",
        "setup_id": None,
        "htf_info": htf_summary,
        "note": f"15M Range: {swing_low:.{digits}f} - {swing_high:.{digits}f}"
    }

    # 5. Eksekusi BUY: Wajib Searah dengan Bias 1H BULLISH
    if htf_bias == "BULLISH" and bullish_mss and bull_fvg:
        fvg_bot, fvg_top, fvg_ce, ob_low, fvg_id = bull_fvg
        entry_price = fvg_top if fvg_top <= eq_15m else fvg_ce
        sl_price = ob_low - (0.15 * atr)
        risk = entry_price - sl_price

        if risk > 0:
            tp1_price = max(swing_high, entry_price + (1.5 * risk))
            # Gunakan 1H Swing High sebagai Draw on Liquidity (TP2) jika memberikan RR >= 2.5
            htf_dol = htf["sh"] if (htf["sh"] and htf["sh"] >= entry_price + (2.5 * risk)) else (entry_price + (3.0 * risk))
            tp2_price = htf_dol

            is_aplus = (htf_zone == "DISCOUNT") and in_kz
            grade = " [A+ ICT 🔥]" if is_aplus else " [B-Grade ⚠️]"

            in_fvg_zone = (fvg_bot <= live_low <= (fvg_top + 0.05 * atr)) and (live_close > sl_price)
            order_type = f"BUY NOW (In FVG) 🟢{grade}" if in_fvg_zone else f"BUY LIMIT (FVG) 🟢{grade}"

            setup = {
                "order": order_type,
                "entry": f"{entry_price:.{digits}f}",
                "sl": f"{sl_price:.{digits}f}",
                "tp1": f"{tp1_price:.{digits}f}",
                "tp2": f"{tp2_price:.{digits}f}",
                "setup_id": f"{pair}_BULL_{fvg_id}",
                "htf_info": htf_summary,
                "note": f"1H Bullish + 15M FVG ({fvg_bot:.{digits}f}-{fvg_top:.{digits}f})"
            }

    # 6. Eksekusi SELL: Wajib Searah dengan Bias 1H BEARISH
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

            in_fvg_zone = ((fvg_bot - 0.05 * atr) <= live_high <= fvg_top) and (live_close < sl_price)
            order_type = f"SELL NOW (In FVG) 🔴{grade}" if in_fvg_zone else f"SELL LIMIT (FVG) 🔴{grade}"

            setup = {
                "order": order_type,
                "entry": f"{entry_price:.{digits}f}",
                "sl": f"{sl_price:.{digits}f}",
                "tp1": f"{tp1_price:.{digits}f}",
                "tp2": f"{tp2_price:.{digits}f}",
                "setup_id": f"{pair}_BEAR_{fvg_id}",
                "htf_info": htf_summary,
                "note": f"1H Bearish + 15M FVG ({fvg_bot:.{digits}f}-{fvg_top:.{digits}f})"
            }

    return setup


def build_actionable_dashboard(data_dict):
    valid_pairs = {k: v for k, v in data_dict.items() if v is not None}
    if not valid_pairs:
        return None

    session = get_current_session()
    kz_status = "ACTIVE 🔥" if is_in_ict_killzone() else "INACTIVE 💤"
    utc_time = datetime.now(timezone.utc).strftime('%H:%M:%S')

    msg = f"⚡ <b>ICT & SMC MTF ACTION PLAN (1H ➔ 15M)</b> ⚡\n"
    msg += f"Sesi: <b>{session}</b> | Killzone: <b>{kz_status}</b>\n"
    msg += f"Waktu Server: <code>{utc_time} UTC</code>\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━\n\n"

    for pair, df in valid_pairs.items():
        cfg = PORTFOLIO[pair]
        curr = df.iloc[-1]
        setup = generate_trade_setup(pair, df, cfg)
        digits = cfg["digits"]

        msg += f"🎯 <b>{pair}</b> | Live Spot: <code>{curr['close']:.{digits}f}</code>\n"
        msg += f"• <b>HTF Bias:</b> <code>{setup['htf_info']}</code>\n"
        msg += f"• <b>Order:</b> <code>{setup['order']}</code>\n"
        msg += f"• <b>Entry:</b> <code>{setup['entry']}</code>\n"
        msg += f"• <b>Stop Loss (SL):</b> <code>{setup['sl']}</code>\n"
        msg += f"• <b>TP1 (Internal):</b> <code>{setup['tp1']}</code>\n"
        msg += f"• <b>TP2 (1H DOL):</b> <code>{setup['tp2']}</code>\n"
        msg += f"• <b>Kondisi:</b> RSI <code>{curr['rsi']:.1f}</code> | <i>{setup['note']}</i>\n"
        msg += "──────────────────────\n"

    msg += "<i>Engine: 1H Dealing Range + 15M Fractal MSS + Displacement FVG</i>"
    return msg


def main():
    start_time = time.time()
    last_15m_report = 0
    triggered_setups = {pair: None for pair in PORTFOLIO}
    weekend_notified = False

    while (time.time() - start_time) < MAX_RUNTIME_SECONDS:
        try:
            if is_forex_market_closed():
                if not weekend_notified:
                    print("Mengambil data penutupan akhir pekan untuk tes koneksi...")
                    data_cache = {pair: fetch_twelve_data(cfg["symbol"]) for pair, cfg in PORTFOLIO.items()}
                    dashboard = build_actionable_dashboard(data_cache)
                    if dashboard:
                        send_telegram(
                            "⏸️ <b>MODE AKHIR PEKAN (PASAR TUTUP)</b>\n"
                            "<i>Bot aktif! Berikut posisi penutupan pasar terakhir sebelum mode hemat kuota:</i>\n\n"
                            + dashboard
                        )
                    weekend_notified = True
                print("Pasar Forex sedang tutup (Akhir Pekan). Tidur 15 menit...")
                time.sleep(900)
                continue

            weekend_notified = False
            now = time.time()
            data_cache = {}

            for pair, cfg in PORTFOLIO.items():
                df = fetch_twelve_data(cfg["symbol"])
                data_cache[pair] = df

                if df is not None:
                    setup = generate_trade_setup(pair, df, cfg)
                    # Cegah spam notifikasi untuk FVG yang sama dengan melacak setup_id unik
                    if "NOW" in setup['order'] and setup['setup_id'] and triggered_setups[pair] != setup['setup_id']:
                        alert_msg = (
                            f"🚨 <b>ICT TRIGGER EXECUTION: {pair}</b> 🚨\n\n"
                            f"HTF Context: <code>{setup['htf_info']}</code>\n"
                            f"Action: <code>{setup['order']}</code>\n"
                            f"Entry: <code>{setup['entry']}</code>\n"
                            f"SL: <code>{setup['sl']}</code>\n"
                            f"TP1: <code>{setup['tp1']}</code> | TP2 (DOL): <code>{setup['tp2']}</code>\n"
                            f"Note: <i>{setup['note']}</i>"
                        )
                        send_telegram(alert_msg)
                        triggered_setups[pair] = setup['setup_id']

            if (now - last_15m_report) >= 900:
                dashboard = build_actionable_dashboard(data_cache)
                if dashboard:
                    send_telegram(dashboard)
                    last_15m_report = now

        except Exception as e:
            print(f"Error loop utama: {e}")

        # Dynamic Throttling: 15 menit saat Sydney/Off-hours, 5 menit saat Tokyo/London/NY
        current_session = get_current_session()
        sleep_duration = OFFHOURS_CHECK_INTERVAL if "Sydney" in current_session else ACTIVE_CHECK_INTERVAL
        time.sleep(sleep_duration)


if __name__ == "__main__":
    main()
