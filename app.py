import streamlit as st, pandas as pd, numpy as np, yfinance as yf, plotly.graph_objects as go
st.set_page_config(page_title="India AI Trading V12",page_icon="🇮🇳",layout="wide")
CAP=100000;FEE=.0005;SLIP=.0005
@st.cache_data(ttl=300)
def load(sym,period,interval):
 x=yf.download(sym,period=period,interval=interval,auto_adjust=False,progress=False)
 if isinstance(x.columns,pd.MultiIndex): x=x.xs(sym,axis=1,level=1,drop_level=True) if sym in x.columns.get_level_values(1) else x.droplevel(1,axis=1)
 x.columns=[str(c) for c in x.columns]
 return x.apply(pd.to_numeric,errors='coerce').dropna()
def feat(x):
 x=x.copy();x['EMA20']=x.Close.ewm(span=20,adjust=False).mean();x['EMA50']=x.Close.ewm(span=50,adjust=False).mean();x['EMA200']=x.Close.ewm(span=200,adjust=False).mean();tr=pd.concat([x.High-x.Low,(x.High-x.Close.shift()).abs(),(x.Low-x.Close.shift()).abs()],axis=1).max(axis=1);x['ATR']=tr.rolling(14).mean();d=x.Close.diff();g=d.clip(lower=0).rolling(14).mean();l=(-d.clip(upper=0)).rolling(14).mean();x['RSI']=100-100/(1+g/l.replace(0,np.nan));x['VWAP']=(x.Close*x.Volume).cumsum()/x.Volume.cumsum();x['VOLMA']=x.Volume.rolling(20).mean();x['ATRpct']=x.ATR/x.Close*100;return x.dropna()
def bt(x,capital=CAP,rr=2.0,risk=.0035):
 cash=capital;pos=0;entry=stop=target=0;peak=capital;dd=0;losses=0;rows=[];eq=[]
 for ts,r in x.iterrows():
  p=float(r.Close);v=cash+pos*p;peak=max(peak,v);dd=max(dd,peak-v);eq.append([ts,v])
  if pos:
   reason='STOP' if p<=stop else 'TARGET' if p>=target else 'EXIT' if r.EMA20<r.EMA50 or p<r.VWAP else None
   if reason:
    ep=p*(1-SLIP);pnl=(ep-entry)*pos-(ep+entry)*pos*FEE;cash+=ep*pos-ep*pos*FEE;rows.append([ts,'SELL',ep,pnl,reason]);losses=losses+1 if pnl<0 else 0;pos=0
  elif losses<2 and r.EMA20>r.EMA50 and r.Close>r.EMA200 and r.Close>r.VWAP and r.Volume>=r.VOLMA and 52<=r.RSI<=66 and .25<=r.ATRpct<=1.5:
   ep=p*(1+SLIP);atr=float(r.ATR)
   if atr<=risk*capital and ep<=cash:cash-=ep;pos=1;entry=ep;stop=ep-atr;target=ep+rr*atr;rows.append([ts,'BUY',ep,0,'ENTRY'])
 if pos:
  ep=float(x.iloc[-1].Close)*(1-SLIP);pnl=(ep-entry)*pos-(ep+entry)*pos*FEE;cash+=ep-ep*FEE;rows.append([x.index[-1],'SELL',ep,pnl,'END'])
 t=pd.DataFrame(rows,columns=['Time','Side','Price','PnL','Reason']);s=t[t.Side=='SELL'];gp=s.loc[s.PnL>0,'PnL'].sum();gl=-s.loc[s.PnL<0,'PnL'].sum();pf=gp/gl if gl else (np.inf if gp>0 else 0);win=(s.PnL>0).mean()*100 if len(s) else 0;return cash,t,win,dd,pf

def oos(df,k=8,rr=2.0,risk=.0035):
 n=len(df);start=int(n*.2);span=max(20,(n-start)//k);rows=[]
 for i in range(k):
  z=df.iloc[start+i*span:min(n,start+(i+1)*span)]
  if len(z)>=20:
   e,t,w,d,p=bt(z,rr=rr,risk=risk);rows.append([i+1,z.index[0],z.index[-1],e-CAP,len(t),w,d,p])
 return pd.DataFrame(rows,columns=['Window','Start','End','P&L','Trades','Win Rate','Max DD','Profit Factor'])
st.title('🇮🇳 India AI Trading Agent — V12 FINAL ROBUSTNESS GATE');st.warning('PAPER TRADING ONLY — no real-money orders.')
sym=st.sidebar.text_input('Symbol','^NSEI');period=st.sidebar.selectbox('Period',['2y','3y','5y'],2);interval=st.sidebar.selectbox('Interval',['1d','1h'],0);raw=load(sym,period,interval)
if raw.empty:st.error('Market data unavailable.');st.stop()
df=feat(raw);r=df.iloc[-1];sig=bool(r.EMA20>r.EMA50 and r.Close>r.EMA200 and r.Close>r.VWAP and r.Volume>=r.VOLMA and 52<=r.RSI<=66 and .25<=r.ATRpct<=1.5)
a,b,c,d=st.columns(4);a.metric('Last',f'{r.Close:,.2f}');b.metric('RSI',f'{r.RSI:.1f}');c.metric('Signal','BUY' if sig else 'HOLD');d.metric('ATR',f'{r.ATR:.2f}')
if st.button('🚀 RUN V12 FINAL GATE',type='primary'):
 end,t,w,dd,pf=bt(df);wf=oos(df);a,b,c,d,e=st.columns(5);a.metric('Ending Value',f'₹{end:,.2f}');b.metric('P&L',f'₹{end-CAP:,.2f}');c.metric('Trades',len(t));d.metric('Win Rate',f'{w:.1f}%');e.metric('Max DD',f'₹{dd:,.2f}');st.metric('Profit Factor','∞' if np.isinf(pf) else f'{pf:.2f}')
 st.subheader('Rolling OOS — Primary');st.dataframe(wf,use_container_width=True)
 if len(wf):
  prof=int((wf['P&L']>0).sum());med=float(wf['P&L'].median());avg=float(wf['P&L'].mean());worst=float(wf['P&L'].min());adequate=int((wf['Trades']>=5).sum());a,b,c,d=st.columns(4);a.metric('Profitable Windows',f'{prof}/{len(wf)}');b.metric('Median OOS P&L',f'₹{med:,.2f}');c.metric('Average OOS P&L',f'₹{avg:,.2f}');d.metric('Worst OOS P&L',f'₹{worst:,.2f}')
 st.subheader('Final Gate');st.write('Gate requires ≥75% profitable OOS windows, positive median OOS P&L, ≥75% windows with ≥5 trades, worst OOS loss better than -₹1,500, and full-period PF ≥ 1.00.')
 passed=len(wf)>0 and prof>=int(np.ceil(.75*len(wf))) and med>0 and adequate>=int(np.ceil(.75*len(wf))) and worst>-1500 and pf>=1
 if passed: st.success('FINAL GATE PASS — still paper-test before any live deployment.')
 else: st.error('FINAL GATE FAIL — do not enable real-money auto orders.')
 st.subheader('Parameter Stability');rows=[]
 for rr,risk,name in [(1.8,.005,'1.8R / 0.50%'),(2.0,.0035,'2.0R / 0.35%'),(2.2,.0025,'2.2R / 0.25%')]:
  e2,t2,w2,d2,p2=bt(df,rr=rr,risk=risk);q=oos(df,8,rr,risk);rows.append([name,e2-CAP,len(t2),w2,d2,p2,int((q['P&L']>0).sum()),float(q['P&L'].median())])
 st.dataframe(pd.DataFrame(rows,columns=['Variant','Full P&L','Trades','Win Rate','Max DD','PF','Profitable OOS','Median OOS P&L']),use_container_width=True)
 fig=go.Figure();fig.add_trace(go.Scatter(x=df.index,y=df.Close,name='Close'));fig.add_trace(go.Scatter(x=df.index,y=df.EMA200,name='EMA200'));st.plotly_chart(fig,use_container_width=True);st.subheader('Trade History');st.dataframe(t,use_container_width=True)
st.info('V12 is a validation gate, not a profit guarantee. No real-money orders.')
