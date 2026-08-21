
import os, numpy as np, pandas as pd, streamlit as st, plotly.graph_objects as go, yfinance as yf
st.set_page_config(page_title="India AI Trading • All-in-One",page_icon="🇮🇳",layout="wide")
CAP=float(os.getenv("STARTING_CAPITAL","100000")); Q=int(os.getenv("TRADE_QTY","1")); SLIP=0.0005; FEE=0.0005
@st.cache_data(ttl=300)
def load(sym,period,interval):
 x=yf.download(sym,period=period,interval=interval,auto_adjust=False,progress=False)
 if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
 return x.dropna()
def indicators(x):
 x=x.copy(); x["EMA20"]=x.Close.ewm(span=20,adjust=False).mean(); x["EMA50"]=x.Close.ewm(span=50,adjust=False).mean()
 tr=pd.concat([x.High-x.Low,(x.High-x.Close.shift()).abs(),(x.Low-x.Close.shift()).abs()],axis=1).max(axis=1); x["ATR"]=tr.rolling(14).mean()
 d=x.Close.diff(); g=d.clip(lower=0).rolling(14).mean(); l=(-d.clip(upper=0)).rolling(14).mean(); x["RSI"]=100-100/(1+g/l.replace(0,np.nan))
 x["VWAP"]=(x.Close*x.Volume).cumsum()/x.Volume.cumsum(); x["VOLMA"]=x.Volume.rolling(20).mean(); return x
def run(x):
 cash=CAP; pos=0; entry=0; stop=target=0; trades=[]; peak=CAP; dd=0; streak=0; halted=False
 for ts,r in x.dropna().iterrows():
  p=float(r.Close); eq=cash+pos*p; peak=max(peak,eq); dd=max(dd,peak-eq)
  if pos:
   reason=None
   if p<=stop: reason="ATR STOP"
   elif p>=target: reason="2R TARGET"
   elif r.EMA20<r.EMA50 or p<r.VWAP: reason="TREND/VWAP EXIT"
   if reason:
    execp=p*(1-SLIP); gross=(execp-entry)*pos; cost=(execp*pos+entry*pos)*FEE; pnl=gross-cost; cash+=execp*pos-cost/2
    trades.append([ts,"SELL",execp,pos,pnl,reason]); streak=streak+1 if pnl<0 else 0; pos=0
    if streak>=3: halted=True
  elif not halted and r.EMA20>r.EMA50 and 52<=r.RSI<=65 and p>r.VWAP and r.Volume>=r.VOLMA:
   risk=1.2*float(r.ATR); execp=p*(1+SLIP)
   if risk>0 and execp*Q<=cash:
    cash-=execp*Q; pos=Q; entry=execp; stop=entry-risk; target=entry+2*risk; trades.append([ts,"BUY",execp,Q,0,"TREND+RSI+VWAP+VOL"])
 if pos:
  p=float(x.iloc[-1].Close); execp=p*(1-SLIP); pnl=(execp-entry)*pos-((execp+entry)*pos*FEE); cash+=execp*pos-((execp*pos)*FEE); trades.append([x.index[-1],"SELL",execp,pos,pnl,"END"])
 t=pd.DataFrame(trades,columns=["Time","Side","Price","Qty","PnL","Reason"]); s=t[t.Side=="SELL"]; gp=s.loc[s.PnL>0,"PnL"].sum(); gl=-s.loc[s.PnL<0,"PnL"].sum()
 return cash,t,(s.PnL>0).mean()*100 if len(s) else 0,dd,gp/gl if gl else np.inf,halted
st.title("🇮🇳 India AI Trading Agent — ALL IN ONE")
st.warning("PAPER TRADING ONLY — no real-money orders.")
sym=st.sidebar.text_input("Symbol","^NSEI"); period=st.sidebar.selectbox("Backtest Period",["6mo","1y","2y"],2); interval=st.sidebar.selectbox("Interval",["1d","1h"],0)
df=indicators(load(sym,period,interval)); r=df.iloc[-1]; sig=r.EMA20>r.EMA50 and 52<=r.RSI<=65 and r.Close>r.VWAP and r.Volume>=r.VOLMA
a,b,c,d,e=st.columns(5); a.metric("Last",f"{r.Close:,.2f}"); b.metric("RSI",f"{r.RSI:.1f}"); c.metric("Signal","BUY" if sig else "HOLD"); d.metric("ATR",f"{r.ATR:.2f}"); e.metric("Capital",f"₹{CAP:,.0f}")
fig=go.Figure(); [fig.add_trace(go.Scatter(x=df.index,y=df[k],name=k)) for k in ["Close","EMA20","EMA50","VWAP"]]; fig.update_layout(height=480); st.plotly_chart(fig,use_container_width=True)
if st.button("🚀 RUN ALL-IN-ONE BACKTEST",type="primary"):
 end,t,w,dd,pf,halt=run(df); pnl=end-CAP
 a,b,c,d,e=st.columns(5); a.metric("Ending Value",f"₹{end:,.2f}"); b.metric("P&L",f"₹{pnl:,.2f}"); c.metric("Trades",len(t)); d.metric("Win Rate",f"{w:.1f}%"); e.metric("Max Drawdown",f"₹{dd:,.2f}")
 st.metric("Profit Factor","∞" if np.isinf(pf) else f"{pf:.2f}")
 st.caption(f"Risk controls: ATR stop, 2R target, 3-loss halt, slippage and estimated fees. Trading halted after 3 consecutive losses: {'YES' if halt else 'NO'}.")
 st.dataframe(t,use_container_width=True)
st.divider(); st.info("For education/testing only. Backtests can be misleading. Live API execution requires broker/exchange controls and applicable SEBI requirements.")
