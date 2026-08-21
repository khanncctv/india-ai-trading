import os,numpy as np,pandas as pd,streamlit as st,plotly.graph_objects as go,yfinance as yf
st.set_page_config(page_title="India AI Trading V5",page_icon="🇮🇳",layout="wide")
CAP=float(os.getenv("STARTING_CAPITAL","100000")); Q=int(os.getenv("TRADE_QTY","1")); SLIP=.0005; FEE=.0005
@st.cache_data(ttl=300)
def get(symbol,period,interval):
 x=yf.download(symbol,period=period,interval=interval,auto_adjust=False,progress=False)
 if isinstance(x.columns,pd.MultiIndex):
  x=x.xs(symbol,axis=1,level=1,drop_level=True) if symbol in x.columns.get_level_values(1) else x.droplevel(1,axis=1)
 x.columns=[str(c) for c in x.columns]
 for c in ["Open","High","Low","Close","Volume"]: 
  if c in x: x[c]=pd.to_numeric(x[c],errors="coerce")
 return x.dropna()
def features(x):
 x=x.copy(); x["EMA20"]=x.Close.ewm(span=20,adjust=False).mean(); x["EMA50"]=x.Close.ewm(span=50,adjust=False).mean()
 tr=pd.concat([x.High-x.Low,(x.High-x.Close.shift()).abs(),(x.Low-x.Close.shift()).abs()],axis=1).max(axis=1); x["ATR"]=tr.rolling(14).mean()
 d=x.Close.diff(); g=d.clip(lower=0).rolling(14).mean(); l=(-d.clip(upper=0)).rolling(14).mean(); x["RSI"]=100-100/(1+g/l.replace(0,np.nan))
 x["VWAP"]=(x.Close*x.Volume).cumsum()/x.Volume.cumsum(); x["VOLMA"]=x.Volume.rolling(20).mean(); return x
def test(x):
 cash=CAP;pos=0;entry=stop=target=0;peak=CAP;dd=0;rows=[];streak=0
 eq=[]; monthly={}
 for ts,r in x.dropna().iterrows():
  p=float(r.Close); eqv=cash+pos*p; peak=max(peak,eqv);dd=max(dd,peak-eqv);eq.append([ts,eqv])
  if pos:
   reason="STOP" if p<=stop else "TARGET" if p>=target else "EXIT" if r.EMA20<r.EMA50 or p<r.VWAP else None
   if reason:
    ep=p*(1-SLIP); pnl=(ep-entry)*pos-(ep+entry)*pos*FEE; cash+=ep*pos-ep*pos*FEE; rows.append([ts,"SELL",ep,pnl,reason]);streak=streak+1 if pnl<0 else 0;pos=0
  elif streak<3 and r.EMA20>r.EMA50 and 52<=r.RSI<=65 and p>r.VWAP and r.Volume>=r.VOLMA:
   risk=1.2*float(r.ATR);ep=p*(1+SLIP)
   if risk>0 and ep*Q<=cash: cash-=ep*Q;pos=Q;entry=ep;stop=ep-risk;target=ep+2*risk;rows.append([ts,"BUY",ep,0,"ENTRY"])
 if pos:
  ep=float(x.iloc[-1].Close)*(1-SLIP);pnl=(ep-entry)*pos-(ep+entry)*pos*FEE;cash+=ep*pos-ep*pos*FEE;rows.append([x.index[-1],"SELL",ep,pnl,"END"])
 t=pd.DataFrame(rows,columns=["Time","Side","Price","PnL","Reason"]);s=t[t.Side=="SELL"];gp=s.loc[s.PnL>0,"PnL"].sum();gl=-s.loc[s.PnL<0,"PnL"].sum()
 equity=pd.DataFrame(eq,columns=["Time","Equity"]).set_index("Time"); ret=(cash-CAP)/CAP*100; win=(s.PnL>0).mean()*100 if len(s) else 0;pf=gp/gl if gl else np.inf
 return cash,t,ret,win,dd,pf,equity
st.title("🇮🇳 India AI Trading Agent — V5 ROBUST TEST");st.warning("PAPER TRADING ONLY — no real-money orders.")
sym=st.sidebar.text_input("Symbol","^NSEI");period=st.sidebar.selectbox("Period",["1y","2y"],1);interval=st.sidebar.selectbox("Interval",["1d","1h"],0)
raw=get(sym,period,interval)
if raw.empty or not {"High","Low","Close","Volume"}.issubset(raw.columns):st.error("Data unavailable. Try ^NSEI / 1d.");st.stop()
df=features(raw);r=df.iloc[-1];sig=bool(r.EMA20>r.EMA50 and 52<=r.RSI<=65 and r.Close>r.VWAP and r.Volume>=r.VOLMA)
a,b,c,d=st.columns(4);a.metric("Last",f"{float(r.Close):,.2f}");b.metric("RSI",f"{float(r.RSI):.1f}");c.metric("Signal","BUY" if sig else "HOLD");d.metric("ATR",f"{float(r.ATR):.2f}")
if st.button("🚀 RUN V5 ROBUST TEST",type="primary"):
 end,t,ret,win,dd,pf,eq=test(df);a,b,c,d,e=st.columns(5);a.metric("Ending Value",f"₹{end:,.2f}");b.metric("P&L",f"₹{end-CAP:,.2f}");c.metric("Trades",len(t));d.metric("Win Rate",f"{win:.1f}%");e.metric("Max DD",f"₹{dd:,.2f}");st.metric("Profit Factor","∞" if np.isinf(pf) else f"{pf:.2f}")
 st.subheader("Equity Curve");fig=go.Figure();fig.add_trace(go.Scatter(x=eq.index.to_numpy(),y=eq.Equity.to_numpy(dtype=float),name="Equity"));st.plotly_chart(fig,use_container_width=True)
 st.subheader("Trade History");st.dataframe(t,use_container_width=True)
 st.caption("V5 includes estimated fees/slippage, ATR risk, 2R target, trend/VWAP exits and 3-loss guard. Backtests are not guarantees.")
