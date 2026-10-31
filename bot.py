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
MAX_RUNTIME_SECONDS = 330 * 60  # 5.5 jam relay
CHECK_INTERVAL_SECONDS = 300    # Cek tiap 5 menit

def send_telegram(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Error Telegram: {e}")

def is_forex_market_closed():
    """Pasar Forex tutup dari Jumat 22:00 UTC hingga Minggu 22:00 UTC"""
    now_utc = datetime.now(timezone.utc)
    weekday = now_utc.weekday() # Senin=0 ... Jumat=4, Sabtu=5, Minggu=6
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

def fetch_twelve_data(symbol):
    """Mengambil 100 candle 15m dari Twelve Data API"""
    if not TWELVE_API_KEY:
        print("Error: TWELVE_API_KEY belum diset di GitHub Secrets!")
        return None

    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": TIMEFRAME,
        "outputsize": 100,
        "apikey": TWELVE_API_KEY
    }
    try:
        res = requests.get(url, params=params, timeout=10).json()
        if "values" not in res:
            print(f"Error Twelve Data ({symbol}): {res.get('message')}")
            return None

        df = pd.DataFrame(res["values"]).iloc[::-1].reset_index(drop=True)
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].astype(float)

        # EMA 50 untuk filter tren intraday
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

        # Wilder's RSI 14 (Akurat sesuai TradingView/MT5)
        delta = df['close'].diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = (100 - (100 / (1 + rs))).fillna(50)

        # Wilder's ATR 14
        hl = df['high'] - df['low']
        hc = (df['high'] - df['close'].shift()).abs()
        lc = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df['atr'] = tr.ewm(alpha=1/14, adjust=False).mean()

        return df
    except Exception as e:
        print(f"Gagal mengambil {symbol}: {e}")
        return None

def generate_trade_setup(pair, df, cfg):
    digits = cfg["digits"]
    live_close = df.iloc[-1]['close']

    # Evaluasi struktur pada candle yang SUDAH CLOSE (iloc[:-1])
    closed_df = df.iloc[:-1].copy()
    last_closed = closed_df.iloc[-1]
    atr = last_closed['atr']
    ema50 = last_closed['ema_50']

    # 1. Swing High & Swing Low dari struktur sebelumnya
    structure_window = closed_df.iloc[-25:-5]
    swing_high = structure_window['high'].max()
    swing_low = structure_window['low'].min()

    # 2. Konfirmasi BOS (Break of Structure) dengan Body Close
    recent_5 = closed_df.iloc[-5:]
    bullish_bos = (recent_5['close'] > swing_high).any() and (last_closed['close'] > ema50)
    bearish_bos = (recent_5['close'] < swing_low).any() and (last_closed['close'] < ema50)

    # 3. Deteksi Unmitigated FVG (Fair Value Gap yang belum ditembus)
    bull_fvg = None
    bear_fvg = None
    for i in range(len(closed_df) - 6, len(closed_df) - 2):
        c1 = closed_df.iloc[i]
        c3 = closed_df.iloc[i + 2]
        subsequent_lows = closed_df.iloc[i + 3:]['low']
        subsequent_highs = closed_df.iloc[i + 3:]['high']

        if (c3['low'] - c1['high']) > (0.15 * atr):
            if subsequent_lows.empty or (subsequent_lows > c1['high']).all():
                bull_fvg = (c1['high'], c3['low'], c1['low'])

        if (c1['low'] - c3['high']) > (0.15 * atr):
            if subsequent_highs.empty or (subsequent_highs < c1['low']).all():
                bear_fvg = (c3['high'], c1['low'], c1['high'])

    setup = {
        "order": "WAIT / NO SETUP ⚪",
        "entry": f"{live_close:.{digits}f}",
        "sl": "-",
        "tp1": "-",
        "tp2": "-",
        "note": f"Menunggu BOS/FVG (Range: {swing_low:.{digits}f} - {swing_high:.{digits}f})"
    }

    # Setup BUY SMC
    if bullish_bos and bull_fvg:
        fvg_bot, fvg_top, struct_low = bull_fvg
        entry_price = fvg_top
        sl_price = min(struct_low, entry_price - (1.2 * atr)) - (0.3 * atr)
        risk = entry_price - sl_price
        tp1_price = entry_price + (1.5 * risk)
        tp2_price = entry_price + (3.0 * risk)

        in_fvg_zone = fvg_bot <= live_close <= (fvg_top + 0.1 * atr)
        order_type = "BUY NOW (In FVG) 🟢" if in_fvg_zone else "BUY LIMIT (FVG) 🟢"

        setup = {
            "order": order_type,
            "entry": f"{entry_price:.{digits}f}",
            "sl": f"{sl_price:.{digits}f}",
            "tp1": f"{tp1_price:.{digits}f}",
            "tp2": f"{tp2_price:.{digits}f}",
            "note": f"Bullish BOS + FVG ({fvg_bot:.{digits}f}-{fvg_top:.{digits}f})"
        }

    # Setup SELL SMC
    elif bearish_bos and bear_fvg:
        fvg_bot, fvg_top, struct_high = bear_fvg
        entry_price = fvg_bot
        sl_price = max(struct_high, entry_price + (1.2 * atr)) + (0.3 * atr)
        risk = sl_price - entry_price
        tp2_price = entry_price - (3.0 * risk)
        tp1_price = entry_price - (1.5 * risk)

        in_fvg_zone = (fvg_bot - 0.1 * atr) <= live_close <= fvg_top
        order_type = "SELL NOW (In FVG) 🔴" if in_fvg_zone else "SELL LIMIT (FVG) 🔴"

        setup = {
            "order": order_type,
            "entry": f"{entry_price:.{digits}f}",
            "sl": f"{sl_price:.{digits}f}",
            "tp1": f"{tp1_price:.{digits}f}",
            "tp2": f"{tp2_price:.{digits}f}",
            "note": f"Bearish BOS + FVG ({fvg_bot:.{digits}f}-{fvg_top:.{digits}f})"
        }

    return setup

def build_actionable_dashboard(data_dict):
    valid_pairs = {k: v for k, v in data_dict.items() if v is not None}
    if not valid_pairs:
        return None

    session = get_current_session()
    utc_time = datetime.now(timezone.utc).strftime('%H:%M:%S')

    msg = f"⚡ <b>SMC FOREX & GOLD ACTION PLAN (15M)</b> ⚡\n"
    msg += f"Sesi: <b>{session}</b> | UTC: <code>{utc_time}</code>\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━\n\n"

    for pair, df in valid_pairs.items():
        cfg = PORTFOLIO[pair]
        curr = df.iloc[-1]
        setup = generate_trade_setup(pair, df, cfg)
        digits = cfg["digits"]

        msg += f"🎯 <b>{pair}</b> | Live Spot: <code>{curr['close']:.{digits}f}</code>\n"
        msg += f"• <b>Order:</b> <code>{setup['order']}</code>\n"
        msg += f"• <b>Entry:</b> <code>{setup['entry']}</code>\n"
        msg += f"• <b>Stop Loss (SL):</b> <code>{setup['sl']}</code>\n"
        msg += f"• <b>Take Profit 1:</b> <code>{setup['tp1']}</code>\n"
        msg += f"• <b>Take Profit 2:</b> <code>{setup['tp2']}</code>\n"
        msg += f"• <b>Kondisi:</b> RSI <code>{curr['rsi']:.1f}</code> | <i>{setup['note']}</i>\n"
        msg += "──────────────────────\n"

    msg += "<i>SMC Engine: Body-Close BOS + Unmitigated FVG</i>"
    return msg

def main():
    start_time = time.time()
    last_15m_report = 0
    last_order_status = {pair: None for pair in PORTFOLIO}
    weekend_notified = False  # Penanda agar di hari libur tetap kirim 1x laporan awal

    while (time.time() - start_time) < MAX_RUNTIME_SECONDS:
        try:
            if is_forex_market_closed():
                if not weekend_notified:
                    # Ambil 1x data penutupan Jumat untuk memastikan Telegram & API jalan
                    data_cache = {pair: fetch_twelve_data(cfg["symbol"]) for pair, cfg in PORTFOLIO.items()}
                    dashboard = build_actionable_dashboard(data_cache)
                    if dashboard:
                        send_telegram("⏸️ <b>MODE AKHIR PEKAN (PASAR TUTUP)</b>\n<i>Berikut posisi penutupan pasar terakhir:</i>\n\n" + dashboard)
                    weekend_notified = True
                print("Pasar Forex sedang tutup (Akhir Pekan). Tidur 15 menit...")
                time.sleep(900)
                continue

            weekend_notified = False
            now = time.time()
            data_cache = {}
