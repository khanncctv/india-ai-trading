import streamlit as st, pandas as pd, numpy as np, yfinance as yf
st.set_page_config(page_title='India AI Trading V19',page_icon='🇮🇳',layout='wide')
CAP=100000
@st.cache_data(ttl=300)
def fetch(sym,period,interval):
 for s in [sym,'^NSEI','NIFTYBEES.NS']:
  try:
   x=yf.download(s,period=period,interval=interval,auto_adjust=False,progress=False,threads=False)
   if isinstance(x.columns,pd.MultiIndex): x=x.droplevel(-1,axis=1)
   x.columns=[str(c) for c in x.columns]
   if all(c in x for c in ['Open','High','Low','Close','Volume']) and len(x)>=200:
    return x.apply(pd.to_numeric,errors='coerce').dropna(),s
  except Exception: pass
 return pd.DataFrame(),''
def feat(x):
 x=x.copy(); x['EMA20']=x.Close.ewm(20,adjust=False).mean(); x['EMA50']=x.Close.ewm(50,adjust=False).mean(); x['EMA200']=x.Close.ewm(200,adjust=False).mean(); x['MOM20']=x.Close.pct_change(20); x['VWAP']=(x.Close*x.Volume).cumsum()/x.Volume.cumsum(); x['VOLMA']=x.Volume.rolling(20).mean(); d=x.Close.diff(); g=d.clip(lower=0).rolling(14).mean(); l=(-d.clip(upper=0)).rolling(14).mean(); x['RSI']=100-100/(1+g/l.replace(0,np.nan)); tr=pd.concat([x.High-x.Low,(x.High-x.Close.shift()).abs(),(x.Low-x.Close.shift()).abs()],axis=1).max(axis=1); x['ATR']=tr.rolling(14).mean(); return x.dropna()
def sig(r):
 edge=(float(r.Close)/float(r.EMA200)-1)*100
 return edge>=1 and r.MOM20>=.01 and r.Close>r.EMA20>r.EMA50>r.EMA200 and r.Close>r.VWAP and r.Volume>=r.VOLMA and 50<=r.RSI<=68
def bt(x,fee):
 cash=CAP; pos=0; entry=0; trades=[]; vals=[]
 for ts,r in x.iterrows():
  p=float(r.Close); vals.append(cash+pos*p)
  if pos and (r.EMA20<r.EMA50 or r.Close<r.VWAP or r.MOM20<0):
   pnl=(p-entry)-(p+entry)*fee; cash+=p-p*fee; trades.append([ts,'SELL',p,pnl]); pos=0
  if not pos and sig(r): cash-=p; pos=1; entry=p; trades.append([ts,'BUY',p,0])
 if pos:
  p=float(x.Close.iloc[-1]); pnl=(p-entry)-(p+entry)*fee; cash+=p-p*fee; trades.append([x.index[-1],'SELL',p,pnl])
 t=pd.DataFrame(trades,columns=['Time','Side','Price','PnL']); s=t[t.Side=='SELL']; gp=s.loc[s.PnL>0,'PnL'].sum(); gl=-s.loc[s.PnL<0,'PnL'].sum(); pf=gp/gl if gl else 0; win=(s.PnL>0).mean()*100 if len(s) else 0
 vv=pd.Series(vals+[cash]); peak=vv.cummax(); dd=float((peak-vv).max()); vol=float(vv.pct_change().std()*np.sqrt(252)) if len(vv)>2 else 0; ret=float(cash/CAP-1); sharpe=ret/vol if vol>0 else 0
 return cash,t,win,pf,dd,sharpe
def oos(x,fee):
 n=len(x); start=int(n*.3); span=max(25,(n-start)//10); rows=[]
 for i in range(10):
  z=x.iloc[start+i*span:min(n,start+(i+1)*span)]
  if len(z)>=25:
   e,t,w,pf,dd,sh=bt(z,fee); rows.append([i+1,e-CAP,len(t),w,pf,dd,sh])
 return pd.DataFrame(rows,columns=['Window','P&L','Trades','Win Rate','Profit Factor','Max DD','Sharpe Proxy'])
st.title('🇮🇳 India AI Trading Agent — V19 RISK-ADJUSTED OOS GATE'); st.warning('PAPER TRADING ONLY — no real-money orders.')
sym=st.sidebar.text_input('Symbol','^NSEI'); period=st.sidebar.selectbox('Period',['3y','5y'],1); interval=st.sidebar.selectbox('Interval',['1d','1h'],0); fee=st.sidebar.slider('Cost / round-trip',0.0,.003,.001,.0005)
raw,src=fetch(sym,period,interval)
if raw.empty: st.error('Market data unavailable. Try 1d.'); st.stop()
df=feat(raw); r=df.iloc[-1]; edge=(float(r.Close)/float(r.EMA200)-1)*100; current=sig(r)
st.caption(f'Data source: {src} • Rows: {len(raw)} • Last bar: {raw.index[-1]}')
a,b,c,d,e=st.columns(5); a.metric('Last',f'{r.Close:,.2f}'); b.metric('RSI',f'{r.RSI:.1f}'); c.metric('Signal','BUY' if current else 'NO TRADE'); d.metric('Trend Edge',f'{edge:.2f}%'); e.metric('20D Momentum',f'{r.MOM20*100:.2f}%')
if st.button('🚀 RUN V19 RESEARCH GATE',type='primary'):
 end,t,w,pf,dd,sh=bt(df,fee); bench=(float(df.Close.iloc[-1])/float(df.Close.iloc[0])-1)*CAP; wf=oos(df,fee); prof=int((wf['P&L']>0).sum()) if len(wf) else 0; med=float(wf['P&L'].median()) if len(wf) else 0; worst=float(wf['P&L'].min()) if len(wf) else 0; oos_med_sh=float(wf['Sharpe Proxy'].median()) if len(wf) else 0
 a,b,c,d,e=st.columns(5); a.metric('Strategy P&L',f'₹{end-CAP:,.2f}'); b.metric('Benchmark P&L',f'₹{bench:,.2f}'); c.metric('Trades',len(t)); d.metric('Win Rate',f'{w:.1f}%'); e.metric('Profit Factor',f'{pf:.2f}')
 st.subheader('10-Window Out-of-Sample'); st.dataframe(wf,use_container_width=True)
 a,b,c,d=st.columns(4); a.metric('Profitable OOS',f'{prof}/10'); b.metric('Median OOS',f'₹{med:,.2f}'); c.metric('Worst OOS',f'₹{worst:,.2f}'); d.metric('Full Sharpe Proxy',f'{sh:.2f}')
 gate=(len(wf)==10 and prof>=8 and med>0 and worst>-1500 and pf>=1.2 and len(t)>=20 and int((wf.Trades>=5).sum())>=8 and sh>0.5)
 if gate: st.success('RESEARCH GATE PASS — continue forward paper validation.')
 else: st.error('RESEARCH GATE FAIL — keep NO TRADE / paper-only. Real-money orders remain OFF.')
 st.subheader('Trade History'); st.dataframe(t,use_container_width=True)
st.info('V19 adds risk-adjusted and OOS consistency gates. No live broker/order functionality is included.')
