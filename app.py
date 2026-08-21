import streamlit as st,pandas as pd,numpy as np,yfinance as yf
st.set_page_config(page_title='India AI Trading V17',page_icon='🇮🇳',layout='wide')
CAP=100000
@st.cache_data(ttl=300)
def fetch(sym,period,interval):
 for s in [sym,'^NSEI','NIFTYBEES.NS']:
  try:
   x=yf.download(s,period=period,interval=interval,auto_adjust=False,progress=False,threads=False)
   if isinstance(x.columns,pd.MultiIndex): x=x.droplevel(-1,axis=1)
   x.columns=[str(c) for c in x.columns]
   if all(c in x for c in ['Open','High','Low','Close','Volume']) and len(x)>=100:return x.apply(pd.to_numeric,errors='coerce').dropna(),s
  except Exception: pass
 return pd.DataFrame(),''
def features(x):
 x=x.copy();x['EMA20']=x.Close.ewm(20,adjust=False).mean();x['EMA50']=x.Close.ewm(50,adjust=False).mean();x['EMA200']=x.Close.ewm(200,adjust=False).mean();x['RET']=x.Close.pct_change();x['VOL20']=x.RET.rolling(20).std();x['MOM20']=x.Close.pct_change(20);x['VWAP']=(x.Close*x.Volume).cumsum()/x.Volume.cumsum();x['VOLMA']=x.Volume.rolling(20).mean();d=x.Close.diff();g=d.clip(lower=0).rolling(14).mean();l=(-d.clip(upper=0)).rolling(14).mean();x['RSI']=100-100/(1+g/l.replace(0,np.nan));tr=pd.concat([x.High-x.Low,(x.High-x.Close.shift()).abs(),(x.Low-x.Close.shift()).abs()],axis=1).max(axis=1);x['ATR']=tr.rolling(14).mean();return x.dropna()
def benchmark_edge(x):
 r=x.iloc[-1];trend=(r.Close/r.EMA200-1)*100;mom=r.MOM20*100;rel=(x.Close.pct_change(20)-x.Close.pct_change(20).rolling(60).mean()).iloc[-1]*100
 return float(trend),float(mom),float(rel)
def backtest(x,fee=.001):
 cash=CAP;pos=0;entry=0;tr=[];peak=CAP;dd=0
 for ts,r in x.iterrows():
  p=float(r.Close);v=cash+pos*p;peak=max(peak,v);dd=max(dd,peak-v)
  trend,mom,rel=benchmark_edge(x.loc[:ts]) if len(x.loc[:ts])>=80 else (0,0,0)
  eligible=(trend>=1 and mom>=1 and rel>=0 and r.EMA20>r.EMA50 and r.Close>r.VWAP and r.Volume>=r.VOLMA and 50<=r.RSI<=68)
  if pos and (r.EMA20<r.EMA50 or r.Close<r.VWAP or mom<0 or trend<0):
   pnl=(p-entry)-(p+entry)*fee;cash+=p-p*fee;tr.append([ts,'SELL',p,pnl]);pos=0
  if not pos and eligible:
   cash-=p;pos=1;entry=p;tr.append([ts,'BUY',p,0])
 if pos:
  p=float(x.Close.iloc[-1]);pnl=(p-entry)-(p+entry)*fee;cash+=p-p*fee;tr.append([x.index[-1],'SELL',p,pnl])
 t=pd.DataFrame(tr,columns=['Time','Side','Price','PnL']);s=t[t.Side=='SELL'];gp=s.loc[s.PnL>0,'PnL'].sum();gl=-s.loc[s.PnL<0,'PnL'].sum();pf=gp/gl if gl else 0;win=(s.PnL>0).mean()*100 if len(s) else 0;return cash,t,win,pf,dd
st.title('🇮🇳 India AI Trading Agent — V17 NO-TRADE BENCHMARK GATE');st.warning('PAPER TRADING ONLY — no real-money orders.')
sym=st.sidebar.text_input('Symbol','^NSEI');period=st.sidebar.selectbox('Period',['3y','5y'],1);interval=st.sidebar.selectbox('Interval',['1d','1h'],0);fee=st.sidebar.slider('Cost / round-trip',0.0,.003,.001,.0005)
raw,src=fetch(sym,period,interval)
if raw.empty:st.error('Market data unavailable. Try 1d.');st.stop()
df=features(raw);trend,mom,rel=benchmark_edge(df);r=df.iloc[-1];eligible=trend>=1 and mom>=1 and rel>=0 and r.EMA20>r.EMA50 and r.Close>r.VWAP and r.Volume>=r.VOLMA and 50<=r.RSI<=68
st.caption(f'Data source: {src} • Rows: {len(raw)} • Last bar: {raw.index[-1]}')
a,b,c,d,e=st.columns(5);a.metric('Last',f'{r.Close:,.2f}');b.metric('RSI',f'{r.RSI:.1f}');c.metric('Signal','BUY' if eligible else 'NO TRADE');d.metric('Trend Edge',f'{trend:.2f}%');e.metric('20D Momentum',f'{mom:.2f}%')
st.info('V17 trades only when trend, momentum, relative edge, volume and trend alignment agree. Otherwise it deliberately stays OUT.')
if st.button('🚀 RUN V17 BENCHMARK GATE',type='primary'):
 end,t,w,pf,dd=backtest(df,fee);bench=(float(df.Close.iloc[-1])/float(df.Close.iloc[0])-1)*CAP;strat=end-CAP
 a,b,c,d,e=st.columns(5);a.metric('Strategy P&L',f'₹{strat:,.2f}');b.metric('Benchmark P&L',f'₹{bench:,.2f}');c.metric('Trades',len(t));d.metric('Win Rate',f'{w:.1f}%');e.metric('Profit Factor',f'{pf:.2f}')
 if strat>bench and pf>=1.2 and len(t)>=20: st.success('BENCHMARK GATE PASS — continue forward paper validation.');
 else: st.error('BENCHMARK GATE FAIL — NO LIVE TRADING. Keep paper-only.')
 st.subheader('Trade History');st.dataframe(t,use_container_width=True)
st.info('No live broker API or real-money order function is included.')
