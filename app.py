
import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf

st.set_page_config(page_title="India AI Trading V2", page_icon="🇮🇳", layout="wide")
SYMBOL=os.getenv("SYMBOL","^NSEI"); CAPITAL=float(os.getenv("STARTING_CAPITAL","100000"))
QTY=int(os.getenv("TRADE_QTY","1")); SL=float(os.getenv("STOP_LOSS_PCT","0.005")); TP=float(os.getenv("TAKE_PROFIT_PCT","0.01"))

@st.cache_data(ttl=300)
def data(symbol,period,interval):
    x=yf.download(symbol,period=period,interval=interval,auto_adjust=False,progress=False)
    if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
    return x.dropna()

def indicators(x):
    x=x.copy(); x["EMA9"]=x.Close.ewm(span=9,adjust=False).mean(); x["EMA21"]=x.Close.ewm(span=21,adjust=False).mean()
    d=x.Close.diff(); g=d.clip(lower=0).rolling(14).mean(); l=(-d.clip(upper=0)).rolling(14).mean()
    x["RSI"]=100-(100/(1+g/l.replace(0,pd.NA)))
    x["VWAP"]=(x.Close*x.Volume).cumsum()/x.Volume.cumsum()
    x["VOL_MA20"]=x.Volume.rolling(20).mean()
    return x

def backtest(x):
    cash=CAPITAL; pos=0; entry=0; trades=[]; peak=CAPITAL; maxdd=0
    for ts,r in x.dropna().iterrows():
        px=float(r.Close)
        if pos==0 and r.EMA9>r.EMA21 and 50<=r.RSI<=68 and px>r.VWAP and r.Volume>=r.VOL_MA20:
            if px*QTY<=cash: cash-=px*QTY; pos=QTY; entry=px; trades.append([ts,"BUY",px,QTY,0,"EMA+RSI+VWAP+VOL"])
        elif pos:
            reason=None
            if px<=entry*(1-SL): reason="STOP LOSS"
            elif px>=entry*(1+TP): reason="TAKE PROFIT"
            elif r.EMA9<r.EMA21 or px<r.VWAP: reason="TREND/VWAP EXIT"
            if reason:
                cash+=px*pos; pnl=(px-entry)*pos; trades.append([ts,"SELL",px,pos,pnl,reason]); pos=0
        equity=cash+(pos*px if pos else 0); peak=max(peak,equity); maxdd=max(maxdd,peak-equity)
    if pos:
        px=float(x.iloc[-1].Close); cash+=px*pos; trades.append([x.index[-1],"SELL",px,pos,(px-entry)*pos,"END"])
    t=pd.DataFrame(trades,columns=["Time","Side","Price","Qty","PnL","Reason"])
    sells=t[t.Side=="SELL"]; wins=(sells.PnL>0).sum(); losses=(sells.PnL<0).sum()
    gross_profit=sells.loc[sells.PnL>0,"PnL"].sum(); gross_loss=-sells.loc[sells.PnL<0,"PnL"].sum()
    ending=cash; return ending,t,(wins/max(len(sells),1))*100,maxdd,(gross_profit/gross_loss if gross_loss else float("inf"))

st.title("🇮🇳 India AI Trading Agent — V2")
st.warning("PAPER TRADING ONLY — no real-money orders.")
symbol=st.sidebar.text_input("Symbol",SYMBOL); period=st.sidebar.selectbox("Period",["3mo","6mo","1y","2y"],2)
if st.sidebar.button("Refresh"): st.cache_data.clear(); st.rerun()
df=indicators(data(symbol,period,"1d"))
last=df.iloc[-1]; buy=last.EMA9>last.EMA21 and 50<=last.RSI<=68 and last.Close>last.VWAP and last.Volume>=last.VOL_MA20
a,b,c,d=st.columns(4); a.metric("Last Price",f"{last.Close:,.2f}"); b.metric("RSI",f"{last.RSI:.1f}"); c.metric("V2 Signal","BUY" if buy else "HOLD"); d.metric("Capital",f"₹{CAPITAL:,.0f}")
fig=go.Figure(); fig.add_trace(go.Scatter(x=df.index,y=df.Close,name="Close")); fig.add_trace(go.Scatter(x=df.index,y=df.EMA9,name="EMA9")); fig.add_trace(go.Scatter(x=df.index,y=df.EMA21,name="EMA21")); fig.add_trace(go.Scatter(x=df.index,y=df.VWAP,name="VWAP")); st.plotly_chart(fig,use_container_width=True)
if st.button("Run V2 Backtest",type="primary"):
    end,t,win,dd,pf=backtest(df); pnl=end-CAPITAL
    a,b,c,d,e=st.columns(5); a.metric("Ending Value",f"₹{end:,.2f}"); b.metric("P&L",f"₹{pnl:,.2f}"); c.metric("Trades",len(t)); d.metric("Win Rate",f"{win:.1f}%"); e.metric("Max Drawdown",f"₹{dd:,.2f}")
    st.metric("Profit Factor", "∞" if pf==float("inf") else f"{pf:.2f}")
    st.dataframe(t,use_container_width=True)
