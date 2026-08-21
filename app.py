import os, numpy as np, pandas as pd, streamlit as st, plotly.graph_objects as go, yfinance as yf
st.set_page_config(page_title="India AI Trading All-in-One",page_icon="🇮🇳",layout="wide")
CAP=float(os.getenv("STARTING_CAPITAL","100000")); QTY=int(os.getenv("TRADE_QTY","1")); SLIP=.0005; FEE=.0005
@st.cache_data(ttl=300)
def load_data(symbol,period,interval):
    x=yf.download(symbol,period=period,interval=interval,auto_adjust=False,progress=False)
    if isinstance(x.columns,pd.MultiIndex):
        x=x.xs(symbol,axis=1,level=1,drop_level=True) if symbol in x.columns.get_level_values(1) else x.droplevel(1,axis=1)
    x.columns=[str(c) for c in x.columns]
    for c in ["Open","High","Low","Close","Volume"]:
        if c in x.columns: x[c]=pd.to_numeric(x[c],errors="coerce")
    return x.dropna()
def ind(x):
    x=x.copy(); x["EMA20"]=x.Close.ewm(span=20,adjust=False).mean(); x["EMA50"]=x.Close.ewm(span=50,adjust=False).mean()
    tr=pd.concat([x.High-x.Low,(x.High-x.Close.shift()).abs(),(x.Low-x.Close.shift()).abs()],axis=1).max(axis=1); x["ATR"]=tr.rolling(14).mean()
    d=x.Close.diff(); g=d.clip(lower=0).rolling(14).mean(); l=(-d.clip(upper=0)).rolling(14).mean(); x["RSI"]=100-100/(1+g/l.replace(0,np.nan))
    x["VWAP"]=(x.Close*x.Volume).cumsum()/x.Volume.cumsum(); x["VOLMA"]=x.Volume.rolling(20).mean(); return x
def bt(x):
    cash=CAP; pos=0; entry=stop=target=0; trades=[]; peak=CAP; dd=0; streak=0; halted=False
    for ts,r in x.dropna().iterrows():
        p=float(r.Close); eq=cash+pos*p; peak=max(peak,eq); dd=max(dd,peak-eq)
        if pos:
            reason="ATR STOP" if p<=stop else "2R TARGET" if p>=target else "TREND/VWAP EXIT" if r.EMA20<r.EMA50 or p<r.VWAP else None
            if reason:
                ep=p*(1-SLIP); pnl=(ep-entry)*pos-((ep+entry)*pos*FEE); cash+=ep*pos-ep*pos*FEE
                trades.append([ts,"SELL",ep,pos,pnl,reason]); streak=streak+1 if pnl<0 else 0; pos=0; halted=streak>=3
        elif not halted and r.EMA20>r.EMA50 and 52<=r.RSI<=65 and p>r.VWAP and r.Volume>=r.VOLMA:
            risk=1.2*float(r.ATR); ep=p*(1+SLIP)
            if risk>0 and ep*QTY<=cash: cash-=ep*QTY; pos=QTY; entry=ep; stop=ep-risk; target=ep+2*risk; trades.append([ts,"BUY",ep,QTY,0,"TREND+RSI+VWAP+VOL"])
    if pos:
        p=float(x.iloc[-1].Close); ep=p*(1-SLIP); pnl=(ep-entry)*pos-((ep+entry)*pos*FEE); cash+=ep*pos-ep*pos*FEE; trades.append([x.index[-1],"SELL",ep,pos,pnl,"END"])
    t=pd.DataFrame(trades,columns=["Time","Side","Price","Qty","PnL","Reason"]); s=t[t.Side=="SELL"]; gp=s.loc[s.PnL>0,"PnL"].sum(); gl=-s.loc[s.PnL<0,"PnL"].sum()
    return cash,t,((s.PnL>0).mean()*100 if len(s) else 0),dd,(gp/gl if gl else np.inf),halted
st.title("🇮🇳 India AI Trading Agent — ALL IN ONE"); st.warning("PAPER TRADING ONLY — no real-money orders.")
sym=st.sidebar.text_input("Symbol","^NSEI"); period=st.sidebar.selectbox("Backtest Period",["6mo","1y","2y"],2); interval=st.sidebar.selectbox("Interval",["1d","1h"],0)
if st.sidebar.button("Refresh data"): st.cache_data.clear(); st.rerun()
raw=load_data(sym,period,interval)
if raw.empty or not {"High","Low","Close","Volume"}.issubset(raw.columns): st.error("Market data unavailable. Try ^NSEI with 1d."); st.stop()
df=ind(raw); r=df.iloc[-1]; sig=bool(r.EMA20>r.EMA50 and 52<=r.RSI<=65 and r.Close>r.VWAP and r.Volume>=r.VOLMA)
a,b,c,d,e=st.columns(5); a.metric("Last",f"{float(r.Close):,.2f}"); b.metric("RSI",f"{float(r.RSI):.1f}"); c.metric("Signal","BUY" if sig else "HOLD"); d.metric("ATR",f"{float(r.ATR):.2f}"); e.metric("Capital",f"₹{CAP:,.0f}")
fig=go.Figure()
for k in ["Close","EMA20","EMA50","VWAP"]:
    fig.add_trace(go.Scatter(x=df.index.to_numpy(), y=df[k].to_numpy(dtype=float), name=k, mode="lines"))
st.plotly_chart(fig,use_container_width=True)
if st.button("🚀 RUN ALL-IN-ONE BACKTEST",type="primary"):
    end,t,w,dd,pf,halt=bt(df); a,b,c,d,e=st.columns(5); a.metric("Ending Value",f"₹{end:,.2f}"); b.metric("P&L",f"₹{end-CAP:,.2f}"); c.metric("Trades",len(t)); d.metric("Win Rate",f"{w:.1f}%"); e.metric("Max Drawdown",f"₹{dd:,.2f}"); st.metric("Profit Factor","∞" if np.isinf(pf) else f"{pf:.2f}"); st.dataframe(t,use_container_width=True)
st.info("Paper trading only. Backtests do not guarantee future performance.")
