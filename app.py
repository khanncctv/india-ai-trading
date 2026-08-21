import streamlit as st, pandas as pd, numpy as np, yfinance as yf, plotly.graph_objects as go
st.set_page_config(page_title='India AI Trading V9',page_icon='🇮🇳',layout='wide')
CAP=100000; Q=1; FEE=.0005; SLIP=.0005
@st.cache_data(ttl=300)
def load(sym,period,interval):
 x=yf.download(sym,period=period,interval=interval,auto_adjust=False,progress=False)
 if isinstance(x.columns,pd.MultiIndex): x=x.xs(sym,axis=1,level=1,drop_level=True) if sym in x.columns.get_level_values(1) else x.droplevel(1,axis=1)
 x.columns=[str(c) for c in x.columns]
 for c in ['Open','High','Low','Close','Volume']:
  if c in x:x[c]=pd.to_numeric(x[c],errors='coerce')
 return x.dropna()
def feat(x):
 x=x.copy(); x['EMA20']=x.Close.ewm(span=20,adjust=False).mean(); x['EMA50']=x.Close.ewm(span=50,adjust=False).mean()
 tr=pd.concat([x.High-x.Low,(x.High-x.Close.shift()).abs(),(x.Low-x.Close.shift()).abs()],axis=1).max(axis=1); x['ATR']=tr.rolling(14).mean()
 d=x.Close.diff(); g=d.clip(lower=0).rolling(14).mean(); l=(-d.clip(upper=0)).rolling(14).mean(); x['RSI']=100-100/(1+g/l.replace(0,np.nan)); x['VWAP']=(x.Close*x.Volume).cumsum()/x.Volume.cumsum(); x['VOLMA']=x.Volume.rolling(20).mean();
 up=x.High.diff(); dn=-x.Low.diff(); plus=up.where((up>dn)&(up>0),0).rolling(14).mean(); minus=dn.where((dn>up)&(dn>0),0).rolling(14).mean(); tm=tr.rolling(14).mean().replace(0,np.nan); x['DI+']=100*plus/tm; x['DI-']=100*minus/tm; x['TS']=(x['DI+']-x['DI-']).abs(); x['ATRpct']=x.ATR/x.Close*100
 return x.dropna()
def bt(x,capital=CAP,qty=Q):
 cash=capital; pos=0; entry=stop=target=0; peak=capital; dd=0; losses=0; rows=[]; eq=[]
 for ts,r in x.iterrows():
  p=float(r.Close); v=cash+pos*p; peak=max(peak,v); dd=max(dd,peak-v); eq.append([ts,v])
  if pos:
   reason='STOP' if p<=stop else 'TARGET' if p>=target else 'EXIT' if r.EMA20<r.EMA50 or p<r.VWAP else None
   if reason:
    ep=p*(1-SLIP); pnl=(ep-entry)*pos-(ep+entry)*pos*FEE; cash+=ep*pos-ep*pos*FEE; rows.append([ts,'SELL',ep,pnl,reason]); losses=losses+1 if pnl<0 else 0; pos=0
  elif losses<3 and r.EMA20>r.EMA50 and r.Close>r.VWAP and r.Volume>=r.VOLMA and r['DI+']>r['DI-'] and r.TS>=8 and .25<=r.ATRpct<=1.5 and 50<=r.RSI<=68:
   risk=1.2*float(r.ATR); ep=p*(1+SLIP)
   if risk*qty<=.0075*capital and ep*qty<=cash: cash-=ep*qty; pos=qty; entry=ep; stop=ep-risk; target=ep+2*risk; rows.append([ts,'BUY',ep,0,'ENTRY'])
 if pos:
  ep=float(x.iloc[-1].Close)*(1-SLIP); pnl=(ep-entry)*pos-(ep+entry)*pos*FEE; cash+=ep*pos-ep*pos*FEE; rows.append([x.index[-1],'SELL',ep,pnl,'END'])
 t=pd.DataFrame(rows,columns=['Time','Side','Price','PnL','Reason']); s=t[t.Side=='SELL']; gp=s.loc[s.PnL>0,'PnL'].sum(); gl=-s.loc[s.PnL<0,'PnL'].sum(); pf=gp/gl if gl else (np.inf if gp>0 else 0); win=(s.PnL>0).mean()*100 if len(s) else 0; e=pd.DataFrame(eq,columns=['Time','Equity']).set_index('Time'); return cash,t,win,dd,pf,e

def rolling_oos(df,windows=6):
 n=len(df); start=int(n*.40); span=max(20,(n-start)//windows); out=[]
 for i in range(windows):
  a=start+i*span; b=min(n,a+span); z=df.iloc[a:b]
  if len(z)<20: continue
  end,t,w,dd,pf,_=bt(z); out.append([i+1,z.index[0],z.index[-1],end-CAP,len(t),w,dd,pf])
 return pd.DataFrame(out,columns=['Window','Start','End','P&L','Trades','Win Rate','Max DD','Profit Factor'])

st.title('🇮🇳 India AI Trading Agent — V9 ROBUSTNESS TEST'); st.warning('PAPER TRADING ONLY — no real-money orders.')
sym=st.sidebar.text_input('Symbol','^NSEI'); period=st.sidebar.selectbox('Period',['2y','3y','5y'],2); interval=st.sidebar.selectbox('Interval',['1d','1h'],0)
raw=load(sym,period,interval)
if raw.empty: st.error('Market data unavailable.'); st.stop()
df=feat(raw); r=df.iloc[-1]
sig=bool(r.EMA20>r.EMA50 and r.Close>r.VWAP and r.Volume>=r.VOLMA and r['DI+']>r['DI-'] and r.TS>=8 and .25<=r.ATRpct<=1.5 and 50<=r.RSI<=68)
a,b,c,d=st.columns(4); a.metric('Last',f'{r.Close:,.2f}'); b.metric('RSI',f'{r.RSI:.1f}'); c.metric('Signal','BUY' if sig else 'HOLD'); d.metric('ATR',f'{r.ATR:.2f}')
if st.button('🚀 RUN V9 ROBUSTNESS TEST',type='primary'):
 end,t,w,dd,pf,eq=bt(df); wf=rolling_oos(df)
 st.subheader('Full Period'); a,b,c,d,e=st.columns(5); a.metric('Ending Value',f'₹{end:,.2f}'); b.metric('P&L',f'₹{end-CAP:,.2f}'); c.metric('Trades',len(t)); d.metric('Win Rate',f'{w:.1f}%'); e.metric('Max DD',f'₹{dd:,.2f}'); st.metric('Profit Factor','∞' if np.isinf(pf) else f'{pf:.2f}')
 st.subheader('Rolling Out-of-Sample Windows'); st.dataframe(wf,use_container_width=True)
 if len(wf):
  profitable=int((wf['P&L']>0).sum()); med=float(wf['P&L'].median()); avg=float(wf['P&L'].mean()); worst=float(wf['P&L'].min()); pf_med=float(wf['Profit Factor'].replace(np.inf,np.nan).median()) if wf['Profit Factor'].notna().any() else 0
  a,b,c,d=st.columns(4); a.metric('Profitable Windows',f'{profitable}/{len(wf)}'); b.metric('Median OOS P&L',f'₹{med:,.2f}'); c.metric('Average OOS P&L',f'₹{avg:,.2f}'); d.metric('Worst OOS P&L',f'₹{worst:,.2f}')
  passed=profitable>=int(np.ceil(len(wf)*.67)) and med>0 and worst>-0.02*CAP and int((wf['Trades']>=5).sum())>=int(np.ceil(len(wf)*.67))
  if passed: st.success('ROBUSTNESS PASS: most OOS windows profitable with adequate trade counts.')
  else: st.error('ROBUSTNESS FAIL: unseen windows are not consistently profitable. Keep real-money trading OFF.')
 fig=go.Figure(); fig.add_trace(go.Scatter(x=eq.index.to_numpy(),y=eq.Equity.to_numpy(dtype=float),name='Equity')); st.plotly_chart(fig,use_container_width=True); st.subheader('Trade History'); st.dataframe(t,use_container_width=True)
st.info('V9 focuses on rolling unseen-window validation, median/worst-window metrics and minimum trade-count checks. No guarantee of future returns.')
