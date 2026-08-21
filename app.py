import streamlit as st,pandas as pd,numpy as np,yfinance as yf
st.set_page_config(page_title='India AI Trading V13',page_icon='🇮🇳',layout='wide')
CAP=100000
@st.cache_data(ttl=300)
def load(s,p,i):
 x=yf.download(s,period=p,interval=i,auto_adjust=False,progress=False)
 if isinstance(x.columns,pd.MultiIndex): x=x.xs(s,axis=1,level=1,drop_level=True) if s in x.columns.get_level_values(1) else x.droplevel(1,axis=1)
 x.columns=[str(c) for c in x.columns]; return x.apply(pd.to_numeric,errors='coerce').dropna()
def feat(x):
 x=x.copy();x['EMA20']=x.Close.ewm(20,adjust=False).mean();x['EMA50']=x.Close.ewm(50,adjust=False).mean();x['EMA200']=x.Close.ewm(200,adjust=False).mean();tr=pd.concat([x.High-x.Low,(x.High-x.Close.shift()).abs(),(x.Low-x.Close.shift()).abs()],axis=1).max(axis=1);x['ATR']=tr.rolling(14).mean();d=x.Close.diff();g=d.clip(lower=0).rolling(14).mean();l=(-d.clip(upper=0)).rolling(14).mean();x['RSI']=100-100/(1+g/l.replace(0,np.nan));x['VWAP']=(x.Close*x.Volume).cumsum()/x.Volume.cumsum();x['VOLMA']=x.Volume.rolling(20).mean();x['ATRpct']=x.ATR/x.Close*100;return x.dropna()
def bt(x,rr=2,risk=.0035,fee=.001):
 cash=CAP;pos=0;entry=stop=target=0;peak=CAP;dd=0;losses=0;tr=[]
 for ts,r in x.iterrows():
  p=float(r.Close);v=cash+pos*p;peak=max(peak,v);dd=max(dd,peak-v)
  if pos:
   reason='STOP' if p<=stop else 'TARGET' if p>=target else 'EXIT' if r.EMA20<r.EMA50 or p<r.VWAP else None
   if reason:
    pnl=(p-entry)*pos-(p+entry)*pos*fee;cash+=p*pos-p*pos*fee;tr.append([ts,'SELL',p,pnl,reason]);losses+=pnl<0;pos=0
  elif losses<2 and r.EMA20>r.EMA50 and r.Close>r.EMA200 and r.Close>r.VWAP and r.Volume>=r.VOLMA and 52<=r.RSI<=66 and .25<=r.ATRpct<=1.5:
   atr=float(r.ATR)
   if atr<=risk*CAP and p<=cash:cash-=p;pos=1;entry=p;stop=p-atr;target=p+rr*atr;tr.append([ts,'BUY',p,0,'ENTRY'])
 if pos:
  p=float(x.Close.iloc[-1]);pnl=(p-entry)-(p+entry)*fee;cash+=p-p*fee;tr.append([x.index[-1],'SELL',p,pnl,'END'])
 t=pd.DataFrame(tr,columns=['Time','Side','Price','PnL','Reason']);s=t[t.Side=='SELL'];gp=s.loc[s.PnL>0,'PnL'].sum();gl=-s.loc[s.PnL<0,'PnL'].sum();pf=gp/gl if gl else (np.inf if gp>0 else 0);win=(s.PnL>0).mean()*100 if len(s) else 0;return cash,t,win,dd,pf
def rolling(df,k=10,rr=2,risk=.0035,fee=.001):
 n=len(df);st=int(n*.2);sp=max(20,(n-st)//k);o=[]
 for i in range(k):
  z=df.iloc[st+i*sp:min(n,st+(i+1)*sp)]
  if len(z)>=20:
   e,t,w,d,p=bt(z,rr,risk,fee);o.append([i+1,z.index[0],z.index[-1],e-CAP,len(t),w,d,p])
 return pd.DataFrame(o,columns=['Window','Start','End','P&L','Trades','Win Rate','Max DD','Profit Factor'])
st.title('🇮🇳 India AI Trading Agent — V13 RESEARCH GATE');st.warning('PAPER TRADING ONLY — no real-money orders.')
sym=st.sidebar.text_input('Symbol','^NSEI');period=st.sidebar.selectbox('Period',['3y','5y'],1);interval=st.sidebar.selectbox('Interval',['1d','1h'],0);fee=st.sidebar.slider('Cost assumption per round-trip',.0005,.003,.001,.0005);raw=load(sym,period,interval)
if raw.empty:st.error('Market data unavailable.');st.stop()
df=feat(raw);r=df.iloc[-1];sig=bool(r.EMA20>r.EMA50 and r.Close>r.EMA200 and r.Close>r.VWAP and r.Volume>=r.VOLMA and 52<=r.RSI<=66 and .25<=r.ATRpct<=1.5);a,b,c,d=st.columns(4);a.metric('Last',f'{r.Close:,.2f}');b.metric('RSI',f'{r.RSI:.1f}');c.metric('Signal','BUY' if sig else 'HOLD');d.metric('ATR',f'{r.ATR:.2f}')
if st.button('🚀 RUN V13 RESEARCH GATE',type='primary'):
 end,t,w,dd,pf=bt(df,fee=fee);wf=rolling(df,10,2,.0035,fee);bench=(float(df.Close.iloc[-1])/float(df.Close.iloc[0])-1)*CAP;a,b,c,d,e=st.columns(5);a.metric('Strategy P&L',f'₹{end-CAP:,.2f}');b.metric('Benchmark P&L',f'₹{bench:,.2f}');c.metric('Trades',len(t));d.metric('Win Rate',f'{w:.1f}%');e.metric('Profit Factor','∞' if np.isinf(pf) else f'{pf:.2f}')
 st.subheader('10-Window Out-of-Sample');st.dataframe(wf,use_container_width=True)
 if len(wf):
  prof=int((wf['P&L']>0).sum());med=float(wf['P&L'].median());avg=float(wf['P&L'].mean());worst=float(wf['P&L'].min());a,b,c,d=st.columns(4);a.metric('Profitable OOS',f'{prof}/{len(wf)}');b.metric('Median OOS',f'₹{med:,.2f}');c.metric('Average OOS',f'₹{avg:,.2f}');d.metric('Worst OOS',f'₹{worst:,.2f}')
 passed=len(wf)>=10 and prof>=8 and med>0 and avg>0 and worst>-1500 and pf>=1 and end-CAP>bench and int((wf['Trades']>=5).sum())>=8
 st.success('RESEARCH PASS — continue paper validation.') if passed else st.error('PAPER-ONLY — research gate failed. Real-money auto orders stay OFF.')
 st.subheader('Parameter + Cost Sensitivity');rows=[]
 for rr in [1.8,2,2.2]:
  for cost in [.0005,.001,.002]:
   e2,t2,w2,d2,p2=bt(df,rr,.0035,cost);q=rolling(df,10,rr,.0035,cost);rows.append([rr,cost,e2-CAP,w2,d2,p2,int((q['P&L']>0).sum()),float(q['P&L'].median())])
 st.dataframe(pd.DataFrame(rows,columns=['R:R','Cost','Full P&L','Win Rate','Max DD','PF','Profitable OOS/10','Median OOS']),use_container_width=True);st.subheader('Trades');st.dataframe(t,use_container_width=True)
