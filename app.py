import streamlit as st, pandas as pd, numpy as np, yfinance as yf
st.set_page_config(page_title='India AI Trading V15',page_icon='🇮🇳',layout='wide')
CAP=100000
@st.cache_data(ttl=300)
def get(s,p,i):
 for q in [s,'^NSEI','NIFTYBEES.NS']:
  try:
   x=yf.download(q,period=p,interval=i,auto_adjust=False,progress=False,threads=False)
   if isinstance(x.columns,pd.MultiIndex): x=x.droplevel(-1,axis=1)
   x.columns=[str(c) for c in x.columns]
   if all(c in x for c in ['Open','High','Low','Close','Volume']) and len(x)>=60:return x.apply(pd.to_numeric,errors='coerce').dropna(),q
  except Exception: pass
 return pd.DataFrame(),''
def feat(x):
 x=x.copy();x['EMA20']=x.Close.ewm(20,adjust=False).mean();x['EMA50']=x.Close.ewm(50,adjust=False).mean();x['EMA200']=x.Close.ewm(200,adjust=False).mean();tr=pd.concat([x.High-x.Low,(x.High-x.Close.shift()).abs(),(x.Low-x.Close.shift()).abs()],axis=1).max(axis=1);x['ATR']=tr.rolling(14).mean();d=x.Close.diff();g=d.clip(lower=0).rolling(14).mean();l=(-d.clip(upper=0)).rolling(14).mean();x['RSI']=100-100/(1+g/l.replace(0,np.nan));x['VWAP']=(x.Close*x.Volume).cumsum()/x.Volume.cumsum();x['VOLMA']=x.Volume.rolling(20).mean();return x.dropna()
def bt(x,fee=.001):
 cash=CAP;pos=0;entry=0;tr=[]
 for ts,r in x.iterrows():
  p=float(r.Close)
  if pos and (r.EMA20<r.EMA50 or p<r.VWAP):
   pnl=(p-entry)-(p+entry)*fee;cash+=p-p*fee;tr.append([ts,'SELL',p,pnl]);pos=0
  edge=(p/float(r.EMA200)-1)*100
  if not pos and edge>=1 and r.EMA20>r.EMA50 and p>r.EMA200 and p>r.VWAP and r.Volume>=r.VOLMA and 52<=r.RSI<=66:
   cash-=p;pos=1;entry=p;tr.append([ts,'BUY',p,0])
 if pos:
  p=float(x.Close.iloc[-1]);cash+=p-p*fee;tr.append([x.index[-1],'SELL',p,(p-entry)-(p+entry)*fee])
 t=pd.DataFrame(tr,columns=['Time','Side','Price','PnL']);s=t[t.Side=='SELL'];gp=s.loc[s.PnL>0,'PnL'].sum();gl=-s.loc[s.PnL<0,'PnL'].sum();return cash,t,gp/gl if gl else 0,(s.PnL>0).mean()*100 if len(s) else 0
st.title('🇮🇳 India AI Trading Agent — V15 BENCHMARK + EDGE FILTER');st.warning('PAPER TRADING ONLY — no real-money orders.')
sym=st.sidebar.text_input('Symbol','^NSEI');period=st.sidebar.selectbox('Period',['3y','5y'],1);interval=st.sidebar.selectbox('Interval',['1d','1h'],0);raw,src=get(sym,period,interval)
if raw.empty:st.error('Market data unavailable.');st.stop()
df=feat(raw);r=df.iloc[-1];edge=(float(r.Close)/float(r.EMA200)-1)*100;sig=edge>=1 and r.EMA20>r.EMA50 and r.Close>r.VWAP and r.Volume>=r.VOLMA and 52<=r.RSI<=66
st.caption(f'Data source: {src} • Rows: {len(raw)} • Last bar: {raw.index[-1]}');a,b,c,d=st.columns(4);a.metric('Last',f'{r.Close:,.2f}');b.metric('RSI',f'{r.RSI:.1f}');c.metric('Signal','BUY' if sig else 'HOLD');d.metric('Trend Edge',f'{edge:.2f}%')
if st.button('RUN V15 RESEARCH GATE',type='primary'):
 e,t,pf,w=bt(df);bench=(float(df.Close.iloc[-1])/float(df.Close.iloc[0])-1)*CAP;a,b,c,d,e5=st.columns(5);a.metric('Strategy P&L',f'₹{e-CAP:,.2f}');b.metric('Benchmark P&L',f'₹{bench:,.2f}');c.metric('Trades',len(t));d.metric('Win Rate',f'{w:.1f}%');e5.metric('Profit Factor',f'{pf:.2f}')
 st.dataframe(t,use_container_width=True);st.error('PAPER-ONLY: research gate not live-trading eligible.')
