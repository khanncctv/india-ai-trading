import streamlit as st, pandas as pd, numpy as np, yfinance as yf, plotly.graph_objects as go
st.set_page_config(page_title='India AI Trading V10',page_icon='🇮🇳',layout='wide')
CAP=100000;Q=1;FEE=.0005;SLIP=.0005
@st.cache_data(ttl=300)
def load(sym,period,interval):
 x=yf.download(sym,period=period,interval=interval,auto_adjust=False,progress=False)
 if isinstance(x.columns,pd.MultiIndex): x=x.xs(sym,axis=1,level=1,drop_level=True) if sym in x.columns.get_level_values(1) else x.droplevel(1,axis=1)
 x.columns=[str(c) for c in x.columns]
 for c in ['Open','High','Low','Close','Volume']:
  if c in x:x[c]=pd.to_numeric(x[c],errors='coerce')
 return x.dropna()
def feat(x):
 x=x.copy();x['EMA20']=x.Close.ewm(span=20,adjust=False).mean();x['EMA50']=x.Close.ewm(span=50,adjust=False).mean();x['EMA200']=x.Close.ewm(span=200,adjust=False).mean();tr=pd.concat([x.High-x.Low,(x.High-x.Close.shift()).abs(),(x.Low-x.Close.shift()).abs()],axis=1).max(axis=1);x['ATR']=tr.rolling(14).mean();d=x.Close.diff();g=d.clip(lower=0).rolling(14).mean();l=(-d.clip(upper=0)).rolling(14).mean();x['RSI']=100-100/(1+g/l.replace(0,np.nan));x['VWAP']=(x.Close*x.Volume).cumsum()/x.Volume.cumsum();x['VOLMA']=x.Volume.rolling(20).mean();return x.dropna()
def bt(x,capital=CAP):
 cash=capital;pos=0;entry=stop=target=0;peak=capital;dd=0;losses=0;rows=[];eq=[]
 for ts,r in x.iterrows():
  p=float(r.Close);v=cash+pos*p;peak=max(peak,v);dd=max(dd,peak-v);eq.append([ts,v])
  if pos:
   reason='STOP' if p<=stop else 'TARGET' if p>=target else 'EXIT' if r.EMA20<r.EMA50 or p<r.VWAP else None
   if reason:
    ep=p*(1-SLIP);pnl=(ep-entry)*pos-(ep+entry)*pos*FEE;cash+=ep*pos-ep*pos*FEE;rows.append([ts,'SELL',ep,pnl,reason]);losses=losses+1 if pnl<0 else 0;pos=0
  elif losses<2 and r.EMA20>r.EMA50 and r.Close>r.EMA200 and r.Close>r.VWAP and r.Volume>=r.VOLMA and 52<=r.RSI<=66:
   risk=1.0*float(r.ATR);ep=p*(1+SLIP)
   if risk*Q<=.005*capital and ep*Q<=cash:cash-=ep*Q;pos=Q;entry=ep;stop=ep-risk;target=ep+1.8*risk;rows.append([ts,'BUY',ep,0,'ENTRY'])
 if pos:
  ep=float(x.iloc[-1].Close)*(1-SLIP);pnl=(ep-entry)*pos-(ep+entry)*pos*FEE;cash+=ep*pos-ep*pos*FEE;rows.append([x.index[-1],'SELL',ep,pnl,'END'])
 t=pd.DataFrame(rows,columns=['Time','Side','Price','PnL','Reason']);s=t[t.Side=='SELL'];gp=s.loc[s.PnL>0,'PnL'].sum();gl=-s.loc[s.PnL<0,'PnL'].sum();pf=gp/gl if gl else (np.inf if gp>0 else 0);win=(s.PnL>0).mean()*100 if len(s) else 0;eq=pd.DataFrame(eq,columns=['Time','Equity']).set_index('Time');return cash,t,win,dd,pf,eq
def rolling(df,k=8):
 n=len(df);start=int(n*.20);span=max(20,(n-start)//k);rows=[]
 for i in range(k):
  a=start+i*span;b=min(n,a+span);z=df.iloc[a:b]
  if len(z)<20:continue
  end,t,w,dd,pf,_=bt(z);rows.append([i+1,z.index[0],z.index[-1],end-CAP,len(t),w,dd,pf])
 return pd.DataFrame(rows,columns=['Window','Start','End','P&L','Trades','Win Rate','Max DD','Profit Factor'])
st.title('🇮🇳 India AI Trading Agent — V10 ROBUST OOS');st.warning('PAPER TRADING ONLY — no real-money orders.')
sym=st.sidebar.text_input('Symbol','^NSEI');period=st.sidebar.selectbox('Period',['2y','3y','5y'],2);interval=st.sidebar.selectbox('Interval',['1d','1h'],0);raw=load(sym,period,interval)
if raw.empty:st.error('Market data unavailable.');st.stop()
df=feat(raw);r=df.iloc[-1];sig=bool(r.EMA20>r.EMA50 and r.Close>r.EMA200 and r.Close>r.VWAP and r.Volume>=r.VOLMA and 52<=r.RSI<=66)
a,b,c,d=st.columns(4);a.metric('Last',f'{r.Close:,.2f}');b.metric('RSI',f'{r.RSI:.1f}');c.metric('Signal','BUY' if sig else 'HOLD');d.metric('ATR',f'{r.ATR:.2f}')
if st.button('🚀 RUN V10 ROBUST OOS',type='primary'):
 end,t,w,dd,pf,eq=bt(df);wf=rolling(df);st.subheader('Full Period');a,b,c,d,e=st.columns(5);a.metric('Ending Value',f'₹{end:,.2f}');b.metric('P&L',f'₹{end-CAP:,.2f}');c.metric('Trades',len(t));d.metric('Win Rate',f'{w:.1f}%');e.metric('Max DD',f'₹{dd:,.2f}');st.metric('Profit Factor','∞' if np.isinf(pf) else f'{pf:.2f}')
 st.subheader('Rolling Out-of-Sample — Primary');st.dataframe(wf,use_container_width=True)
 if len(wf):
  prof=int((wf['P&L']>0).sum());median=float(wf['P&L'].median());avg=float(wf['P&L'].mean());worst=float(wf['P&L'].min());adequate=int((wf['Trades']>=5).sum());a,b,c,d=st.columns(4);a.metric('Profitable Windows',f'{prof}/{len(wf)}');b.metric('Median OOS P&L',f'₹{median:,.2f}');c.metric('Average OOS P&L',f'₹{avg:,.2f}');d.metric('Worst OOS P&L',f'₹{worst:,.2f}');passed=prof>=int(np.ceil(.75*len(wf))) and median>0 and adequate>=int(np.ceil(.75*len(wf))) and worst>-1500
  if passed:st.success('ROBUSTNESS PASS')
  else:st.error('ROBUSTNESS FAIL — keep real-money trading OFF')
 fig=go.Figure();fig.add_trace(go.Scatter(x=eq.index,y=eq.Equity,name='Equity'));st.plotly_chart(fig,use_container_width=True);st.subheader('Trades');st.dataframe(t,use_container_width=True)
st.info('V10 is stricter than V9: longer-term trend confirmation, lower risk, fewer consecutive-loss allowance, and 8 rolling OOS windows. Validation only.')
