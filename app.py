
import os, numpy as np, pandas as pd, streamlit as st, plotly.graph_objects as go, yfinance as yf
st.set_page_config(page_title="India AI Trading V3",page_icon="🇮🇳",layout="wide")
CAP=float(os.getenv("STARTING_CAPITAL","100000")); Q=int(os.getenv("TRADE_QTY","1"))
@st.cache_data(ttl=300)
def load(sym,period,interval):
 x=yf.download(sym,period=period,interval=interval,auto_adjust=False,progress=False)
 if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
 return x.dropna()
def ind(x):
 x=x.copy(); x["EMA20"]=x.Close.ewm(span=20,adjust=False).mean(); x["EMA50"]=x.Close.ewm(span=50,adjust=False).mean()
 tr=pd.concat([x.High-x.Low,(x.High-x.Close.shift()).abs(),(x.Low-x.Close.shift()).abs()],axis=1).max(axis=1); x["ATR"]=tr.rolling(14).mean()
 d=x.Close.diff(); g=d.clip(lower=0).rolling(14).mean(); l=(-d.clip(upper=0)).rolling(14).mean(); x["RSI"]=100-100/(1+g/l.replace(0,np.nan))
 x["VWAP"]=(x.Close*x.Volume).cumsum()/x.Volume.cumsum(); x["VOLMA"]=x.Volume.rolling(20).mean(); return x
def bt(x):
 cash=CAP; pos=0; entry=0; stop=target=0; trades=[]; equity=[]; peak=CAP; dd=0; loss_streak=0
 for ts,r in x.dropna().iterrows():
  p=float(r.Close); eq=cash+pos*p; peak=max(peak,eq); dd=max(dd,peak-eq); equity.append((ts,eq))
  if pos:
   reason=None
   if p<=stop: reason="ATR STOP"
   elif p>=target: reason="2R TARGET"
   elif r.EMA20<r.EMA50: reason="TREND EXIT"
   if reason:
    cash+=p*pos; pnl=(p-entry)*pos; trades.append([ts,"SELL",p,pnl,reason]); loss_streak=loss_streak+1 if pnl<0 else 0; pos=0
  elif loss_streak<3 and r.EMA20>r.EMA50 and 52<=r.RSI<=65 and p>r.VWAP and r.Volume>=r.VOLMA:
   risk=float(r.ATR)*1.2; stop=p-risk; target=p+2*risk
   if risk>0 and p*Q<=cash: cash-=p*Q; pos=Q; entry=p; trades.append([ts,"BUY",p,0,"TREND+RSI+VWAP+VOL"])
 if pos:
  p=float(x.iloc[-1].Close); cash+=p*pos; trades.append([x.index[-1],"SELL",p,(p-entry)*pos,"END"])
 t=pd.DataFrame(trades,columns=["Time","Side","Price","PnL","Reason"]); s=t[t.Side=="SELL"]
 gp=s.loc[s.PnL>0,"PnL"].sum(); gl=-s.loc[s.PnL<0,"PnL"].sum()
 win=(s.PnL>0).mean()*100 if len(s) else 0; pf=gp/gl if gl else np.inf
 return cash,t,win,dd,pf
st.title("🇮🇳 India AI Trading Agent — V3")
st.warning("PAPER TRADING ONLY — no real-money orders.")
sym=st.sidebar.text_input("Symbol","^NSEI"); period=st.sidebar.selectbox("Period",["6mo","1y","2y"],2)
df=ind(load(sym,period,"1d")); r=df.iloc[-1]
signal= r.EMA20>r.EMA50 and 52<=r.RSI<=65 and r.Close>r.VWAP and r.Volume>=r.VOLMA
a,b,c,d=st.columns(4); a.metric("Last",f"{r.Close:,.2f}"); b.metric("RSI",f"{r.RSI:.1f}"); c.metric("V3 Signal","BUY" if signal else "HOLD"); d.metric("ATR",f"{r.ATR:.2f}")
fig=go.Figure([go.Scatter(x=df.index,y=df.Close,name="Close"),go.Scatter(x=df.index,y=df.EMA20,name="EMA20"),go.Scatter(x=df.index,y=df.EMA50,name="EMA50"),go.Scatter(x=df.index,y=df.VWAP,name="VWAP")]); fig.update_layout(height=500); st.plotly_chart(fig,use_container_width=True)
if st.button("Run V3 Backtest",type="primary"):
 end,t,w,dd,pf=bt(df); pnl=end-CAP
 a,b,c,d,e=st.columns(5); a.metric("Ending Value",f"₹{end:,.2f}"); b.metric("P&L",f"₹{pnl:,.2f}"); c.metric("Trades",len(t)); d.metric("Win Rate",f"{w:.1f}%"); e.metric("Max Drawdown",f"₹{dd:,.2f}")
 st.metric("Profit Factor","∞" if np.isinf(pf) else f"{pf:.2f}"); st.dataframe(t,use_container_width=True)
