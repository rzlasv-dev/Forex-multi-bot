import time
import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ----------------- KONFIGURASI -----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

PORTFOLIO = {
    "XAU/USD": {"symbol": "GC=F", "pip": 0.1, "digits": 2, "name": "GOLD"},
    "EUR/USD": {"symbol": "EURUSD=X", "pip": 0.0001, "digits": 5, "name": "EUR/USD"},
    "USD/JPY": {"symbol": "JPY=X", "pip": 0.01, "digits": 3, "name": "USD/JPY"}
}

TIMEFRAME = "15m"
MAX_RUNTIME_SECONDS = 330 * 60  # 5.5 jam relay
CHECK_INTERVAL_SECONDS = 60

def send_telegram(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Token atau Chat ID kosong!")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID, 
        "text": message, 
        "parse_mode": "HTML"
    }
    try:
        res = requests.post(url, json=payload, timeout=10)
        if res.status_code != 200:
            print(f"Error respon Telegram: {res.text}")
    except Exception as e:
        print(f"Error kirim Telegram: {e}")

def is_market_open():
    now = datetime.now(timezone.utc)
    weekday = now.weekday()
    hour = now.hour
    if weekday == 5 and hour >= 22:
        return False
    if weekday == 6 and hour < 21:
        return False
    return True

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

def fetch_data(pair, cfg):
    try:
        ticker = yf.Ticker(cfg["symbol"])
        df = ticker.history(period="3d", interval=TIMEFRAME)
        if df.empty:
            return None
        df = df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})

        # EMA 9, 21, 50
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

        # RSI 14
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # ATR 14
        hl = df['high'] - df['low']
        hc = (df['high'] - df['close'].shift()).abs()
        lc = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()

        return df
    except Exception as e:
        print(f"Gagal mengambil {pair}: {e}")
        return None

def generate_trade_setup(pair, df, cfg):
    curr = df.iloc[-1]
    digits = cfg["digits"]
    close = curr['close']
    ema9 = curr['ema_9']
    ema21 = curr['ema_21']
    atr = curr['atr']
    rsi = curr['rsi']

    setup = {
        "order": "WAIT / NO SETUP ⚪",
        "entry": f"{close:.{digits}f}",
        "sl": "-",
        "tp1": "-",
        "tp2": "-",
        "note": "Konsolidasi harga"
    }

    # Setup SELL
    if ema9 < ema21 and rsi < 55:
        entry_price = max(close, ema9)
        sl_price = entry_price + (1.5 * atr)
        tp1_price = entry_price - (1.5 * atr)
        tp2_price = entry_price - (3.0 * atr)

        order_type = "SELL NOW 🔴" if close <= ema9 and rsi < 40 else "SELL LIMIT (Pullback) 🔴"
        setup = {
            "order": order_type,
            "entry": f"{entry_price:.{digits}f}",
            "sl": f"{sl_price:.{digits}f}",
            "tp1": f"{tp1_price:.{digits}f}",
            "tp2": f"{tp2_price:.{digits}f}",
            "note": "Bearish momentum. Rejeksi EMA 9."
        }

    # Setup BUY
    elif ema9 > ema21 and rsi > 45:
        entry_price = min(close, ema9)
        sl_price = entry_price - (1.5 * atr)
        tp1_price = entry_price + (1.5 * atr)
        tp2_price = entry_price + (3.0 * atr)

        order_type = "BUY NOW 🟢" if close >= ema9 and rsi > 60 else "BUY LIMIT (Pullback) 🟢"
        setup = {
            "order": order_type,
            "entry": f"{entry_price:.{digits}f}",
            "sl": f"{sl_price:.{digits}f}",
            "tp1": f"{tp1_price:.{digits}f}",
            "tp2": f"{tp2_price:.{digits}f}",
            "note": "Bullish momentum. Rebound EMA 9."
        }

    return setup

def build_actionable_dashboard(data_dict):
    session = get_current_session()
    utc_time = datetime.now(timezone.utc).strftime('%H:%M:%S')

    msg = f"⚡ <b>FOREX DAN GOLD ACTION PLAN (15M)</b> ⚡\n"
    msg += f"Session: <b>{session}</b> | UTC: <code>{utc_time}</code>\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━\n\n"

    for pair, df in data_dict.items():
        if df is None:
            continue
        cfg = PORTFOLIO[pair]
        curr = df.iloc[-1]
        setup = generate_trade_setup(pair, df, cfg)
        digits = cfg["digits"]

        msg += f"🎯 <b>{pair}</b> | Price: <code>{curr['close']:.{digits}f}</code>\n"
        msg += f"• <b>Order:</b> <code>{setup['order']}</code>\n"
        msg += f"• <b>Entry Area:</b> <code>{setup['entry']}</code>\n"
        msg += f"• <b>Stop Loss (SL):</b> <code>{setup['sl']}</code>\n"
        msg += f"• <b>Take Profit 1:</b> <code>{setup['tp1']}</code>\n"
        msg += f"• <b>Take Profit 2:</b> <code>{setup['tp2']}</code>\n"
        msg += f"• <b>Kondisi:</b> RSI <code>{curr['rsi']:.1f}</code> | <i>{setup['note']}</i>\n"
        msg += "──────────────────────\n"

    msg += "<i>Trigger real-time aktif scan tiap 60 detik.</i>"
    return msg

def main():
    start_time = time.time()
    last_15m_report = 0
    last_order_status = {pair: None for pair in PORTFOLIO}

    if not is_market_open():
        send_telegram("⏸ <b>Market Forex sedang Weekend.</b> Bot standby.")
        return

    # Kirim salam pembuka versi TERBARU
    send_telegram("🚀 <b>Bot Forex Multi-Pair Aktif.</b> Membaca data dan setup...")

    while (time.time() - start_time) < MAX_RUNTIME_SECONDS:
        try:
            if not is_market_open():
                break

            now = time.time()
            data_cache = {}

            # Ambil data semua pair
            for pair, cfg in PORTFOLIO.items():
                df = fetch_data(pair, cfg)
                data_cache[pair] = df

                # Trigger Sinyal Instan
                if df is not None:
                    setup = generate_trade_setup(pair, df, cfg)
                    if "NOW" in setup['order'] and last_order_status[pair] != setup['order']:
                        alert_msg = (
                            f"🚨 <b>TRIGGER EXECUTION: {pair}</b> 🚨\n\n"
                            f"Action: <code>{setup['order']}</code>\n"
                            f"Price: <code>{setup['entry']}</code>\n"
                            f"SL: <code>{setup['sl']}</code> | TP1: <code>{setup['tp1']}</code>"
                        )
                        send_telegram(alert_msg)
                        last_order_status[pair] = setup['order']
                    elif "NOW" not in setup['order']:
                        last_order_status[pair] = None

            # Kirim Dashboard: LANGSUNG DI DETIK PERTAMA, lalu tiap 15 menit
            if (now - last_15m_report) >= 900:
                dashboard = build_actionable_dashboard(data_cache)
                send_telegram(dashboard)
                last_15m_report = now

        except Exception as e:
            print(f"Error loop utama: {e}")

        time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
