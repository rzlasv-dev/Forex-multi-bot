import time
import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ----------------- KONFIGURASI TELEGRAM -----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Konfigurasi Multi-Pair
# XAUUSD menggunakan Gold Futures (GC=F), EURUSD (EURUSD=X), USDJPY (JPY=X)
PORTFOLIO = {
    "XAU/USD": {"ticker": "GC=F", "pip": 0.1, "digits": 2, "name": "GOLD"},
    "EUR/USD": {"ticker": "EURUSD=X", "pip": 0.0001, "digits": 5, "name": "EUR/USD"},
    "USD/JPY": {"ticker": "JPY=X", "pip": 0.01, "digits": 3, "name": "USD/JPY"}
}

TIMEFRAME = "15m"
MAX_RUNTIME_SECONDS = 330 * 60  # 5.5 jam per relay
CHECK_INTERVAL_SECONDS = 60     # Cek sinyal tiap 1 menit

def send_telegram(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Telegram error: {e}")

def is_market_open():
    """Tutup: Sabtu 22:00 UTC - Minggu 21:00 UTC"""
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
        sessions.append("Asia/Tokyo 🇯🇵")
    if 7 <= hour < 16:
        sessions.append("London 🇬🇧")
    if 13 <= hour < 22:
        sessions.append("New York 🇺🇸")
    return " & ".join(sessions) if sessions else "Off-hours / Sydney 🇦🇺"

def fetch_and_analyze(symbol_key, config):
    ticker = yf.Ticker(config["ticker"])
    df = ticker.history(period="5d", interval=TIMEFRAME)
    if df.empty:
        return None

    df = df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
    
    # EMA 50 & 200
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # RSI 14
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # ATR (Pips)
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['atr_pips'] = (tr.rolling(14).mean()) / config["pip"]

    return df

def check_trigger_setup(pair_name, df, digits):
    """Deteksi Pullback Rejection di EMA 50 sesuai arah Major Trend (EMA 200)"""
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    
    # BUY SETUP
    if curr['close'] > curr['ema_200']:
        if prev['low'] <= prev['ema_50'] and curr['close'] > curr['ema_50'] and 40 <= curr['rsi'] <= 60:
            return f"🚨 *TRIGGER BUY SETUP: {pair_name}* 🟢\nDynamic Support Rejection (EMA 50)\nPrice: `{curr['close']:.{digits}f}` | RSI: `{curr['rsi']:.1f}`"
            
    # SELL SETUP
    if curr['close'] < curr['ema_200']:
        if prev['high'] >= prev['ema_50'] and curr['close'] < curr['ema_50'] and 40 <= curr['rsi'] <= 60:
            return f"🚨 *TRIGGER SELL SETUP: {pair_name}* 🔴\nDynamic Resistance Rejection (EMA 50)\nPrice: `{curr['close']:.{digits}f}` | RSI: `{curr['rsi']:.1f}`"

    return None

def build_15m_dashboard(data_dict):
    session = get_current_session()
    utc_time = datetime.now(timezone.utc).strftime('%H:%M:%S')
    
    msg = f"🏛 *FOREX & GOLD DASHBOARD (15M)*\n"
    msg += f"Session: `{session}` | UTC: `{utc_time}`\n"
    msg += "─" * 28 + "\n\n"

    for pair, df in data_dict.items():
        if df is None:
            continue
        last = df.iloc[-1]
        cfg = PORTFOLIO[pair]
        digits = cfg["digits"]

        # Trend Bias
        if last['close'] > last['ema_50'] > last['ema_200']:
            bias = "BULLISH 🟢"
        elif last['close'] < last['ema_50'] < last['ema_200']:
            bias = "BEARISH 🔴"
        else:
            bias = "SIDEWAYS ⚪"

        msg += f"📌 *{pair}* (`{last['close']:.{digits}f}`)\n"
        msg += f"• Bias: {bias}\n"
        msg += f"• RSI: `{last['rsi']:.1f}` | Volatility: `±{last['atr_pips']:.1f} Pips`\n"
        msg += f"• Key EMA: `50: {last['ema_50']:.{digits}f}` | `200: {last['ema_200']:.{digits}f}`\n\n"

    msg += "_Scanner trigger aktif memeriksa tiap 60 detik._"
    return msg

def main():
    start_time = time.time()
    last_15m_report = 0
    sent_triggers = {pair: None for pair in PORTFOLIO}

    if not is_market_open():
        send_telegram("⏸ *Market Forex sedang Weekend (Tutup).* Bot istirahat.")
        return

    send_telegram("🚀 *Forex Multi-Bot Aktif* memantau: `XAU/USD`, `EUR/USD`, `USD/JPY`.")

    while (time.time() - start_time) < MAX_RUNTIME_SECONDS:
        try:
            if not is_market_open():
                break

            now = time.time()
            data_cache = {}

            # Loop scan data ke-3 pair
            for pair, cfg in PORTFOLIO.items():
                df = fetch_and_analyze(pair, cfg)
                data_cache[pair] = df

                # 1. Evaluasi Trigger Instan
                if df is not None:
                    trigger_msg = check_trigger_setup(pair, df, cfg["digits"])
                    if trigger_msg and sent_triggers[pair] != trigger_msg:
                        send_telegram(trigger_msg)
                        sent_triggers[pair] = trigger_msg
                    elif not trigger_msg:
                        sent_triggers[pair] = None

            # 2. Kirim Pesan Dashboard Tiap 15 Menit (900 detik)
            if (now - last_15m_report) >= 900:
                dashboard = build_15m_dashboard(data_cache)
                send_telegram(dashboard)
                last_15m_report = now

        except Exception as e:
            print(f"Error loop utama: {e}")

        time.sleep(CHECK_INTERVAL_SECONDS)

    print("Siklus runner selesai. Melakukan handover relay.")

if __name__ == "__main__":
    main()
