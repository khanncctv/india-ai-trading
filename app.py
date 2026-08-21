import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf

st.set_page_config(page_title="India AI Trading Cloud", page_icon="🇮🇳", layout="wide")

SYMBOL = os.getenv("SYMBOL", "^NSEI")
CAPITAL = float(os.getenv("STARTING_CAPITAL", "100000"))
QTY = int(os.getenv("TRADE_QTY", "1"))
SL = float(os.getenv("STOP_LOSS_PCT", "0.005"))
TP = float(os.getenv("TAKE_PROFIT_PCT", "0.01"))

@st.cache_data(ttl=300)
def get_data(symbol, period, interval):
    df = yf.download(symbol, period=period, interval=interval,
                     auto_adjust=False, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).title() for c in df.columns]
    return df.dropna()

def add_indicators(df):
    x = df.copy()
    x["EMA9"] = x["Close"].ewm(span=9, adjust=False).mean()
    x["EMA21"] = x["Close"].ewm(span=21, adjust=False).mean()
    delta = x["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, pd.NA)
    x["RSI"] = 100 - (100 / (1 + rs))
    return x

def run_backtest(df):
    cash, position, entry = CAPITAL, 0, 0
    trades = []
    for ts, r in df.dropna().iterrows():
        price = float(r["Close"])
        if position == 0:
            if r["EMA9"] > r["EMA21"] and 50 <= r["RSI"] <= 70:
                cost = price * QTY
                if cost <= cash:
                    cash -= cost
                    position, entry = QTY, price
                    trades.append([ts, "BUY", price, QTY, 0, "EMA+RSI"])
        else:
            reason = None
            if price <= entry * (1-SL): reason = "STOP LOSS"
            elif price >= entry * (1+TP): reason = "TAKE PROFIT"
            elif r["EMA9"] < r["EMA21"]: reason = "TREND EXIT"
            if reason:
                cash += price * position
                pnl = (price-entry)*position
                trades.append([ts, "SELL", price, position, pnl, reason])
                position = 0
    if position:
        price = float(df.iloc[-1]["Close"])
        cash += price * position
        trades.append([df.index[-1], "SELL", price, position, (price-entry)*position, "END"])
    return cash, pd.DataFrame(trades, columns=["Time","Side","Price","Qty","PnL","Reason"])

st.title("🇮🇳 India AI Trading Agent")
st.caption("Cloud paper-trading dashboard • No real-money orders")

st.sidebar.header("Market")
symbol = st.sidebar.text_input("Symbol", SYMBOL)
period = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
interval = st.sidebar.selectbox("Interval", ["1d", "1h"], index=0)

if st.sidebar.button("Refresh data"):
    st.cache_data.clear()
    st.rerun()

df = get_data(symbol, period, interval)
if df.empty:
    st.error("Market data could not be loaded. Try again later or choose another symbol.")
    st.stop()

df = add_indicators(df)
last = df.iloc[-1]
buy = bool(last["EMA9"] > last["EMA21"] and 50 <= last["RSI"] <= 70)

c1,c2,c3,c4 = st.columns(4)
c1.metric("Last Price", f"{float(last['Close']):,.2f}")
c2.metric("RSI", f"{float(last['RSI']):.2f}")
c3.metric("Signal", "BUY" if buy else "HOLD")
c4.metric("Paper Capital", f"₹{CAPITAL:,.0f}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close"))
fig.add_trace(go.Scatter(x=df.index, y=df["EMA9"], name="EMA 9"))
fig.add_trace(go.Scatter(x=df.index, y=df["EMA21"], name="EMA 21"))
fig.update_layout(height=520, margin=dict(l=10,r=10,t=40,b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Backtest")
if st.button("Run Backtest", type="primary"):
    ending, trades = run_backtest(df)
    pnl = ending - CAPITAL
    a,b,c = st.columns(3)
    a.metric("Ending Value", f"₹{ending:,.2f}")
    b.metric("P&L", f"₹{pnl:,.2f}")
    c.metric("Trades", len(trades))
    st.dataframe(trades, use_container_width=True)

st.divider()
st.info("Educational paper-trading MVP. Do not use these signals as investment advice. Real trading requires broker/exchange integration, authentication, risk controls and compliance.")
