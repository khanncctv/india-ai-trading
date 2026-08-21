import streamlit as st, pandas as pd, numpy as np, yfinance as yf
st.set_page_config(page_title='India AI Trading V20',page_icon='🇮🇳',layout='wide')
CAP=100000.0
@st.cache_data(ttl=300)
def load(sym,period,interval):
 for s in [sym,'^NSEI','NIFTYBEES.NS']:
  try:
   x=yf.download(s,period=period,interval=interval,auto_adjust=False,progress=False,threads=False)
   if isinstance(x.columns,pd.MultiIndex): x=x.droplevel(-1,axis=1)
   x.columns=[str(c) for c in x.columns]
   if all(c in x.columns for c in ['Open','High','Low','Close','Volume']) and len(x)>250:return x.apply(pd.to_numeric,errors='coerce').dropna(),s
  except: pass
 return pd.DataFrame(),''
def ind(x):
 x=x.copy(); c=x.Close
 x['EMA20']=c.ewm(20,adjust=False).mean();x['EMA50']=c.ewm(50,adjust=False).mean();x['EMA200']=c.ewm(200,adjust=False).mean();x['MOM20']=c.pct_change(20);x['MOM60']=c.pct_change(60);x['VWAP']=(c*x.Volume).cumsum()/x.Volume.cumsum();x['VOLMA']=x.Volume.rolling(20).mean();d=c.diff();g=d.clip(lower=0).rolling(14).mean();l=(-d.clip(upper=0)).rolling(14).mean();x['RSI']=100-100/(1+g/l.replace(0,np.nan));tr=pd.concat([x.High-x.Low,(x.High-x.Close.shift()).abs(),(x.Low-x.Close.shift()).abs()],axis=1).max(axis=1);x['ATR']=tr.rolling(14).mean();x['ATRpct']=x.ATR/c;return x.dropna()
def buy(r): return bool((r.Close/r.EMA200-1)>=.01 and r.MOM20>=.01 and r.MOM60>=.03 and r.Close>r.EMA20>r.EMA50>r.EMA200 and r.Close>r.VWAP and r.Volume>=r.VOLMA and 50<=r.RSI<=68 and r.ATRpct<.035)
def bt(x,fee,slip):
 cash=CAP;pos=0;entry=0;rows=[];eq=[]
 for ts,r in x.iterrows():
  p=float(r.Close);eq.append(cash+pos*p)
  if pos and (r.EMA20<r.EMA50 or r.Close<r.VWAP or r.MOM20<0 or r.MOM60<0):
   q=p*(1-slip); pnl=(q-entry)-(q+entry)*fee;cash+=q-q*fee;rows.append([ts,'SELL',q,pnl]);pos=0
  if not pos and buy(r):
   q=p*(1+slip);cash-=q+q*fee;pos=1;entry=q;rows.append([ts,'BUY',q,0])
 if pos:
  p=float(x.Close.iloc[-1]);q=p*(1-slip);pnl=(q-entry)-(q+entry)*fee;cash+=q-q*fee;rows.append([x.index[-1],'SELL',q,pnl])
 t=pd.DataFrame(rows,columns=['Time','Side','Price','PnL']);s=t[t.Side=='SELL'];gp=s.loc[s.PnL>0,'PnL'].sum();gl=-s.loc[s.PnL<0,'PnL'].sum();pf=gp/gl if gl else 0;win=(s.PnL>0).mean()*100 if len(s) else 0;v=pd.Series(eq+[cash]);dd=float((v.cummax()-v).max());r=v.pct_change().dropna();sh=float(r.mean()/r.std()*np.sqrt(252)) if r.std()>0 else 0;return cash,t,win,pf,dd,sh
def oos(x,fee,slip):
 n=len(x);start=int(n*.35);span=max(30,(n-start)//10);out=[]
 for i in range(10):
  z=x.iloc[start+i*span:min(n,start+(i+1)*span)]
  if len(z)>=30:
   e,t,w,pf,dd,sh=bt(z,fee,slip);out.append([i+1,e-CAP,len(t[t.Side=='SELL']),w,pf,dd,sh])
 return pd.DataFrame(out,columns=['Window','P&L','Trades','Win Rate','Profit Factor','Max DD','Sharpe'])
st.title('🇮🇳 India AI Trading Agent — V20 RESEARCH GRADE');st.warning('PAPER TRADING ONLY — no real-money orders.')
sym=st.sidebar.text_input('Symbol','^NSEI');period=st.sidebar.selectbox('Period',['3y','5y'],1);interval=st.sidebar.selectbox('Interval',['1d','1h'],0);fee=st.sidebar.slider('Round-trip cost',0.,.004,.001,.0005);slip=st.sidebar.slider('Slippage',0.,.003,.0005,.0005)
raw,src=load(sym,period,interval)
if raw.empty:st.error('Market data unavailable. Try ^NSEI / 5y / 1d.');st.stop()
df=ind(raw);r=df.iloc[-1];edge=(float(r.Close/r.EMA200)-1)*100;sig=buy(r);st.caption(f'Data source: {src} • Rows: {len(raw)} • Last bar: {raw.index[-1]}');a,b,c,d,e=st.columns(5);a.metric('Last',f'{r.Close:,.2f}');b.metric('RSI',f'{r.RSI:.1f}');c.metric('Signal','BUY' if sig else 'NO TRADE');d.metric('Trend Edge',f'{edge:.2f}%');e.metric('20D Momentum',f'{r.MOM20*100:.2f}%')
if st.button('🚀 RUN V20 RESEARCH AUDIT',type='primary'):
 end,t,w,pf,dd,sh=bt(df,fee,slip);wf=oos(df,fee,slip);bench=(float(df.Close.iloc[-1])/float(df.Close.iloc[0])-1)*CAP;prof=int((wf['P&L']>0).sum()) if len(wf) else 0;med=float(wf['P&L'].median()) if len(wf) else 0;worst=float(wf['P&L'].min()) if len(wf) else 0
 a,b,c,d,e,f=st.columns(6);a.metric('Strategy P&L',f'₹{end-CAP:,.2f}');b.metric('Benchmark P&L',f'₹{bench:,.2f}');c.metric('Trades',int((t.Side=='SELL').sum()));d.metric('Win Rate',f'{w:.1f}%');e.metric('Profit Factor',f'{pf:.2f}');f.metric('Sharpe',f'{sh:.2f}');st.subheader('Rolling Out-of-Sample — Locked');st.dataframe(wf,use_container_width=True);a,b,c,d=st.columns(4);a.metric('Profitable OOS',f'{prof}/{len(wf)}');b.metric('Median OOS',f'₹{med:,.2f}');c.metric('Worst OOS',f'₹{worst:,.2f}');d.metric('Max DD',f'₹{dd:,.2f}');gate=(len(wf)==10 and prof>=8 and med>0 and worst>-1500 and int((t.Side=='SELL').sum())>=20 and pf>=1.2 and sh>=.5 and end-CAP>0)
 if gate:st.success('RESEARCH GATE PASS — forward paper validation required.')
 else:st.error('RESEARCH GATE FAIL — NO TRADE / PAPER ONLY. Real-money orders OFF.')
 st.dataframe(t,use_container_width=True)
