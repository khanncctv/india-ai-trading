import os, numpy as np, pandas as pd, streamlit as st, plotly.graph_objects as go, yfinance as yf
st.set_page_config(page_title="India AI Trading V6",page_icon="🇮🇳",layout="wide")
CAP=float(os.getenv("STARTING_CAPITAL","100000")); Q=int(os.getenv("TRADE_QTY","1")); SLIP=.0005; FEE=.0005
@st.cache_data(ttl=300)
def get(sym,period,interval):
    x=yf.download(sym,period=period,interval=interval,auto_adjust=False,progress=False)
    if isinstance(x.columns,pd.MultiIndex):
        x=x.xs(sym,axis=1,level=1,drop_level=True) if sym in x.columns.get_level_values(1) else x.droplevel(1,axis=1)
    x.columns=[str(c) for c in x.columns]
    for c in ["Open","High","Low","Close","Volume"]:
        if c in x: x[c]=pd.to_numeric(x[c],errors="coerce")
    return x.dropna()
def feat(x):
    x=x.copy(); x["EMA20"]=x.Close.ewm(span=20,adjust=False).mean(); x["EMA50"]=x.Close.ewm(span=50,adjust=False).mean()
    tr=pd.concat([x.High-x.Low,(x.High-x.Close.shift()).abs(),(x.Low-x.Close.shift()).abs()],axis=1).max(axis=1); x["ATR"]=tr.rolling(14).mean()
    d=x.Close.diff(); g=d.clip(lower=0).rolling(14).mean(); l=(-d.clip(upper=0)).rolling(14).mean(); x["RSI"]=100-100/(1+g/l.replace(0,np.nan))
    x["VWAP"]=(x.Close*x.Volume).cumsum()/x.Volume.cumsum(); x["VOLMA"]=x.Volume.rolling(20).mean(); return x.dropna()
def run(x,capital=CAP):
    cash=capital; pos=0; entry=stop=target=0; peak=capital; dd=0; streak=0; rows=[]; eq=[]
    for ts,r in x.iterrows():
        p=float(r.Close); equity=cash+pos*p; peak=max(peak,equity); dd=max(dd,peak-equity); eq.append([ts,equity])
        if pos:
            reason="STOP" if p<=stop else "TARGET" if p>=target else "EXIT" if r.EMA20<r.EMA50 or p<r.VWAP else None
            if reason:
                ep=p*(1-SLIP); pnl=(ep-entry)*pos-((ep+entry)*pos*FEE); cash+=ep*pos-ep*pos*FEE; rows.append([ts,"SELL",ep,pnl,reason]); streak=streak+1 if pnl<0 else 0; pos=0
        elif streak<3 and r.EMA20>r.EMA50 and 52<=r.RSI<=65 and r.Close>r.VWAP and r.Volume>=r.VOLMA:
            risk=1.2*float(r.ATR); ep=p*(1+SLIP)
            if risk>0 and ep*Q<=cash: cash-=ep*Q; pos=Q; entry=ep; stop=ep-risk; target=ep+2*risk; rows.append([ts,"BUY",ep,0,"ENTRY"])
    if pos:
        ep=float(x.iloc[-1].Close)*(1-SLIP); pnl=(ep-entry)*pos-((ep+entry)*pos*FEE); cash+=ep*pos-ep*pos*FEE; rows.append([x.index[-1],"SELL",ep,pnl,"END"])
    t=pd.DataFrame(rows,columns=["Time","Side","Price","PnL","Reason"]); s=t[t.Side=="SELL"]; gp=s.loc[s.PnL>0,"PnL"].sum(); gl=-s.loc[s.PnL<0,"PnL"].sum()
    return cash,t,(s.PnL>0).mean()*100 if len(s) else 0,dd,(gp/gl if gl else np.inf),pd.DataFrame(eq,columns=["Time","Equity"]).set_index("Time")
st.title("🇮🇳 India AI Trading Agent — V6 FINAL ROBUST")
st.warning("PAPER TRADING ONLY — no real-money orders.")
sym=st.sidebar.text_input("Symbol","^NSEI"); period=st.sidebar.selectbox("Period",["2y","3y","5y"],0); interval=st.sidebar.selectbox("Interval",["1d","1h"],0)
raw=get(sym,period,interval)
if raw.empty or not {"High","Low","Close","Volume"}.issubset(raw.columns): st.error("Data unavailable. Try ^NSEI / 1d."); st.stop()
df=feat(raw); r=df.iloc[-1]; sig=bool(r.EMA20>r.EMA50 and 52<=r.RSI<=65 and r.Close>r.VWAP and r.Volume>=r.VOLMA)
a,b,c,d=st.columns(4); a.metric("Last",f"{float(r.Close):,.2f}"); b.metric("RSI",f"{float(r.RSI):.1f}"); c.metric("Signal","BUY" if sig else "HOLD"); d.metric("ATR",f"{float(r.ATR):.2f}")
if st.button("🚀 RUN V6 FINAL ROBUST TEST",type="primary"):
    n=len(df); split=int(n*.70); train=df.iloc[:split]; test=df.iloc[split:]
    tr_end,trades,tr_win,tr_dd,tr_pf,tr_eq=run(train)
    te_end,te_trades,te_win,te_dd,te_pf,te_eq=run(test,CAP)
    all_end,all_trades,all_win,all_dd,all_pf,all_eq=run(df)
    st.subheader("Out-of-Sample (30%) — Primary Result")
    a,b,c,d,e=st.columns(5); a.metric("Ending Value",f"₹{te_end:,.2f}"); b.metric("P&L",f"₹{te_end-CAP:,.2f}"); c.metric("Trades",len(te_trades)); d.metric("Win Rate",f"{te_win:.1f}%"); e.metric("Max DD",f"₹{te_dd:,.2f}")
    st.metric("Profit Factor","∞" if np.isinf(te_pf) else f"{te_pf:.2f}")
    st.subheader("In-Sample (70%)")
    a,b,c,d,e=st.columns(5); a.metric("P&L",f"₹{tr_end-CAP:,.2f}"); b.metric("Trades",len(trades)); c.metric("Win Rate",f"{tr_win:.1f}%"); d.metric("Max DD",f"₹{tr_dd:,.2f}"); e.metric("Profit Factor","∞" if np.isinf(tr_pf) else f"{tr_pf:.2f}")
    st.subheader("Full Period")
    st.write(f"Ending Value ₹{all_end:,.2f} • P&L ₹{all_end-CAP:,.2f} • Trades {len(all_trades)} • Win Rate {all_win:.1f}% • Max DD ₹{all_dd:,.2f} • Profit Factor {'∞' if np.isinf(all_pf) else f'{all_pf:.2f}'}")
    fig=go.Figure(); fig.add_trace(go.Scatter(x=te_eq.index.to_numpy(),y=te_eq.Equity.to_numpy(dtype=float),name="Out-of-Sample Equity")); st.plotly_chart(fig,use_container_width=True)
    st.subheader("Out-of-Sample Trades"); st.dataframe(te_trades,use_container_width=True)
    if te_end>CAP and te_pf>1: st.success("Out-of-sample result is profitable. This is encouraging, but not proof of future performance.")
    else: st.error("Out-of-sample result is not strong enough. Do not use real money.")
st.info("V6 separates 70% in-sample and 30% out-of-sample data. Fees/slippage and risk rules are included. Backtests cannot guarantee future returns.")
