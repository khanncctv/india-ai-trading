import streamlit as st, pandas as pd, numpy as np, yfinance as yf, plotly.graph_objects as go
st.set_page_config(page_title='India AI Trading V11',page_icon='🇮🇳',layout='wide')
CAP=100000; FEE=.0005; SLIP=.0005
@st.cache_data(ttl=300)
def load(sym,period,interval):
 x=yf.download(sym,period=period,interval=interval,auto_adjust=False,progress=False)
 if isinstance(x.columns,pd.MultiIndex): x=x.xs(sym,axis=1,level=1,drop_level=True) if sym in x.columns.get_level_values(1) else x.droplevel(1,axis=1)
 x.columns=[str(c) for c in x.columns]
 for c in ['Open','High','Low','Close','Volume']:
  if c in x:x[c]=pd.to_numeric(x[c],errors='coerce')
 return x.dropna()
def feat(x):
 x=x.copy(); x['EMA20']=x.Close.ewm(span=20,adjust=False).mean(); x['EMA50']=x.Close.ewm(span=50,adjust=False).mean(); x['EMA200']=x.Close.ewm(span=200,adjust=False).mean()
 tr=pd.concat([x.High-x.Low,(x.High-x.Close.shift()).abs(),(x.Low-x.Close.shift()).abs()],axis=1).max(axis=1); x['ATR']=tr.rolling(14).mean()
 d=x.Close.diff(); g=d.clip(lower=0).rolling(14).mean(); l=(-d.clip(upper=0)).rolling(14).mean(); x['RSI']=100-100/(1+g/l.replace(0,np.nan)); x['VWAP']=(x.Close*x.Volume).cumsum()/x.Volume.cumsum(); x['VOLMA']=x.Volume.rolling(20).mean(); x['ATRpct']=x.ATR/x.Close*100; return x.dropna()
def bt(x,variant=0):
 cash=CAP; pos=0; entry=stop=target=0; peak=CAP; dd=0; losses=0; rows=[]; eq=[]; risk=.005 if variant<2 else .0035; rr=1.8 if variant==0 else 2.0
 for ts,r in x.iterrows():
  p=float(r.Close); v=cash+pos*p; peak=max(peak,v); dd=max(dd,peak-v); eq.append([ts,v])
  if pos:
   reason='STOP' if p<=stop else 'TARGET' if p>=target else 'EXIT' if r.EMA20<r.EMA50 or p<r.VWAP else None
   if reason:
    ep=p*(1-SLIP); pnl=(ep-entry)*pos-(ep+entry)*pos*FEE; cash+=ep*pos-ep*pos*FEE; rows.append([ts,'SELL',ep,pnl,reason]); losses=losses+1 if pnl<0 else 0; pos=0
  elif losses<2 and r.EMA20>r.EMA50 and r.Close>r.EMA200 and r.Close>r.VWAP and r.Volume>=r.VOLMA and 52<=r.RSI<=66 and .25<=r.ATRpct<=1.5:
   ep=p*(1+SLIP); risk_unit=float(r.ATR)
   if risk_unit<=risk*CAP and ep<=cash: cash-=ep; pos=1; entry=ep; stop=ep-risk_unit; target=ep+rr*risk_unit; rows.append([ts,'BUY',ep,0,'ENTRY'])
 if pos:
  ep=float(x.iloc[-1].Close)*(1-SLIP); pnl=(ep-entry)*pos-(ep+entry)*pos*FEE; cash+=ep-ep*FEE; rows.append([x.index[-1],'SELL',ep,pnl,'END'])
 t=pd.DataFrame(rows,columns=['Time','Side','Price','PnL','Reason']); s=t[t.Side=='SELL']; gp=s.loc[s.PnL>0,'PnL'].sum(); gl=-s.loc[s.PnL<0,'PnL'].sum(); pf=gp/gl if gl else (np.inf if gp>0 else 0); win=(s.PnL>0).mean()*100 if len(s) else 0; e=pd.DataFrame(eq,columns=['Time','Equity']).set_index('Time'); return cash,t,win,dd,pf,e
def windows(df,k=8,v=0):
 n=len(df); start=int(n*.2); span=max(20,(n-start)//k); out=[]
 for i in range(k):
  z=df.iloc[start+i*span:min(n,start+(i+1)*span)]
  if len(z)>=20:
   e,t,w,d,p,_=bt(z,v); out.append([i+1,z.index[0],z.index[-1],e-CAP,len(t),w,d,p])
 return pd.DataFrame(out,columns=['Window','Start','End','P&L','Trades','Win Rate','Max DD','Profit Factor'])
st.title('🇮🇳 India AI Trading Agent — V11 ALL IN ONE'); st.warning('PAPER TRADING ONLY — no real-money orders.')
sym=st.sidebar.text_input('Symbol','^NSEI'); period=st.sidebar.selectbox('Period',['2y','3y','5y'],2); interval=st.sidebar.selectbox('Interval',['1d','1h'],0); raw=load(sym,period,interval)
if raw.empty: st.error('Market data unavailable.'); st.stop()
df=feat(raw); r=df.iloc[-1]; sig=bool(r.EMA20>r.EMA50 and r.Close>r.EMA200 and r.Close>r.VWAP and r.Volume>=r.VOLMA and 52<=r.RSI<=66 and .25<=r.ATRpct<=1.5)
a,b,c,d=st.columns(4); a.metric('Last',f'{r.Close:,.2f}'); b.metric('RSI',f'{r.RSI:.1f}'); c.metric('Signal','BUY' if sig else 'HOLD'); d.metric('ATR',f'{r.ATR:.2f}')
if st.button('🚀 RUN V11 FULL ROBUSTNESS',type='primary'):
 end,t,w,dd,pf,eq=bt(df); wf=windows(df)
 a,b,c,d,e=st.columns(5); a.metric('Ending Value',f'₹{end:,.2f}'); b.metric('P&L',f'₹{end-CAP:,.2f}'); c.metric('Trades',len(t)); d.metric('Win Rate',f'{w:.1f}%'); e.metric('Max DD',f'₹{dd:,.2f}'); st.metric('Profit Factor','∞' if np.isinf(pf) else f'{pf:.2f}')
 st.subheader('Rolling OOS — Primary'); st.dataframe(wf,use_container_width=True)
 if len(wf):
  prof=int((wf['P&L']>0).sum()); med=float(wf['P&L'].median()); avg=float(wf['P&L'].mean()); worst=float(wf['P&L'].min()); adequate=int((wf['Trades']>=5).sum()); a,b,c,d=st.columns(4); a.metric('Profitable Windows',f'{prof}/{len(wf)}'); b.metric('Median OOS P&L',f'₹{med:,.2f}'); c.metric('Average OOS P&L',f'₹{avg:,.2f}'); d.metric('Worst OOS P&L',f'₹{worst:,.2f}'); passed=prof>=int(np.ceil(.75*len(wf))) and med>0 and adequate>=int(np.ceil(.75*len(wf))) and worst>-1500; st.success('ROBUSTNESS PASS') if passed else st.error('ROBUSTNESS FAIL — keep real-money trading OFF')
 st.subheader('Parameter Stability'); rows=[]
 for v,name in [(0,'Base 0.5% / 1.8R'),(1,'0.5% / 2R'),(2,'0.35% / 2R')]:
  e2,t2,w2,d2,p2,_=bt(df,v); q=windows(df,8,v); rows.append([name,e2-CAP,len(t2),w2,d2,p2,int((q['P&L']>0).sum()),float(q['P&L'].median())])
 st.dataframe(pd.DataFrame(rows,columns=['Variant','Full P&L','Trades','Win Rate','Max DD','PF','Profitable OOS','Median OOS P&L']),use_container_width=True)
 fig=go.Figure(); fig.add_trace(go.Scatter(x=eq.index,y=eq.Equity,name='Equity')); st.plotly_chart(fig,use_container_width=True); st.subheader('Trade History'); st.dataframe(t,use_container_width=True)
st.info('V11 combines regime filters, risk controls, rolling OOS, parameter stability and automatic PASS/FAIL. Validation only; no real-money orders.')
