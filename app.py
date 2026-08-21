import streamlit as st, pandas as pd, numpy as np, yfinance as yf, plotly.graph_objects as go
st.set_page_config(page_title="India AI Trading V8.1",page_icon="🇮🇳",layout="wide")
CAP=100000; QTY=1; FEE=.0005; SLIP=.0005
@st.cache_data(ttl=300)
def data(sym,period,interval):
 x=yf.download(sym,period=period,interval=interval,auto_adjust=False,progress=False)
 if isinstance(x.columns,pd.MultiIndex): x=x.xs(sym,axis=1,level=1,drop_level=True) if sym in x.columns.get_level_values(1) else x.droplevel(1,axis=1)
 x.columns=[str(c) for c in x.columns]
 for c in ['Open','High','Low','Close','Volume']:
  if c in x:x[c]=pd.to_numeric(x[c],errors='coerce')
 return x.dropna()
def feat(x):
 x=x.copy();x['EMA20']=x.Close.ewm(span=20,adjust=False).mean();x['EMA50']=x.Close.ewm(span=50,adjust=False).mean();tr=pd.concat([x.High-x.Low,(x.High-x.Close.shift()).abs(),(x.Low-x.Close.shift()).abs()],axis=1).max(axis=1);x['ATR']=tr.rolling(14).mean();d=x.Close.diff();g=d.clip(lower=0).rolling(14).mean();l=(-d.clip(upper=0)).rolling(14).mean();x['RSI']=100-100/(1+g/l.replace(0,np.nan));x['VWAP']=(x.Close*x.Volume).cumsum()/x.Volume.cumsum();x['VOLMA']=x.Volume.rolling(20).mean();up=x.High.diff();dn=-x.Low.diff();plus=up.where((up>dn)&(up>0),0).rolling(14).mean();minus=dn.where((dn>up)&(dn>0),0).rolling(14).mean();tm=tr.rolling(14).mean().replace(0,np.nan);x['DI+']=100*plus/tm;x['DI-']=100*minus/tm;x['TS']=(x['DI+']-x['DI-']).abs();x['ATRpct']=x.ATR/x.Close*100;return x.dropna()
def bt(x,capital=CAP):
 cash=capital;pos=0;entry=stop=target=0;peak=capital;dd=0;losses=0;rows=[];eq=[]
 for ts,r in x.iterrows():
  p=float(r.Close);v=cash+pos*p;peak=max(peak,v);dd=max(dd,peak-v);eq.append([ts,v])
  if pos:
   reason='STOP' if p<=stop else 'TARGET' if p>=target else 'EXIT' if r.EMA20<r.EMA50 or p<r.VWAP else None
   if reason:
    ep=p*(1-SLIP);pnl=(ep-entry)*pos-(ep+entry)*pos*FEE;cash+=ep*pos-ep*pos*FEE;rows.append([ts,'SELL',ep,pnl,reason]);losses=losses+1 if pnl<0 else 0;pos=0
  elif losses<3 and r.EMA20>r.EMA50 and r.Close>r.VWAP and r.Volume>=r.VOLMA and r['DI+']>r['DI-'] and r.TS>=8 and .25<=r.ATRpct<=1.5 and 50<=r.RSI<=68:
   risk=1.2*float(r.ATR);ep=p*(1+SLIP)
   if risk*QTY<=.0075*capital and ep*QTY<=cash:cash-=ep*QTY;pos=QTY;entry=ep;stop=ep-risk;target=ep+2*risk;rows.append([ts,'BUY',ep,0,'ENTRY'])
 if pos:
  ep=float(x.iloc[-1].Close)*(1-SLIP);pnl=(ep-entry)*pos-(ep+entry)*pos*FEE;cash+=ep*pos-ep*pos*FEE;rows.append([x.index[-1],'SELL',ep,pnl,'END'])
 t=pd.DataFrame(rows,columns=['Time','Side','Price','PnL','Reason']);s=t[t.Side=='SELL'];gp=s.loc[s.PnL>0,'PnL'].sum();gl=-s.loc[s.PnL<0,'PnL'].sum();pf=gp/gl if gl else (np.inf if gp else 0);win=(s.PnL>0).mean()*100 if len(s) else 0;return cash,t,win,dd,pf,pd.DataFrame(eq,columns=['Time','Equity']).set_index('Time')
st.title('🇮🇳 India AI Trading Agent — V8.1 OOS + WALK-FORWARD');st.warning('PAPER TRADING ONLY — no real-money orders.')
sym=st.sidebar.text_input('Symbol','^NSEI');period=st.sidebar.selectbox('Period',['2y','3y','5y'],2);interval=st.sidebar.selectbox('Interval',['1d','1h'],0);raw=data(sym,period,interval)
if raw.empty:st.error('Market data unavailable.');st.stop()
df=feat(raw);r=df.iloc[-1];sig=bool(r.EMA20>r.EMA50 and r.Close>r.VWAP and r.Volume>=r.VOLMA and r['DI+']>r['DI-'] and r.TS>=8 and .25<=r.ATRpct<=1.5 and 50<=r.RSI<=68)
a,b,c,d=st.columns(4);a.metric('Last',f'{r.Close:,.2f}');b.metric('RSI',f'{r.RSI:.1f}');c.metric('Signal','BUY' if sig else 'HOLD');d.metric('ATR',f'{r.ATR:.2f}')
if st.button('🚀 RUN V8.1 VALIDATION',type='primary'):
 n=len(df); cut=int(n*.7); ins=df.iloc[:cut];oos=df.iloc[cut:];ie,it,iw,id_,ip,iq=bt(ins);oe,ot,ow,od,op,oq=bt(oos);fe,ft,fw,fd,fp,fq=bt(df)
 st.subheader('Out-of-Sample 30% — Primary');a,b,c,d,e=st.columns(5);a.metric('Ending Value',f'₹{oe:,.2f}');b.metric('P&L',f'₹{oe-CAP:,.2f}');c.metric('Trades',len(ot));d.metric('Win Rate',f'{ow:.1f}%');e.metric('Max DD',f'₹{od:,.2f}');st.metric('Profit Factor','∞' if np.isinf(op) else f'{op:.2f}')
 st.subheader('In-Sample 70%');st.write(f'P&L ₹{ie-CAP:,.2f} • Trades {len(it)} • Win Rate {iw:.1f}% • Max DD ₹{id_:,.2f} • Profit Factor {"∞" if np.isinf(ip) else f"{ip:.2f}"}')
 st.subheader('Full Period');st.write(f'Ending Value ₹{fe:,.2f} • P&L ₹{fe-CAP:,.2f} • Trades {len(ft)} • Win Rate {fw:.1f}% • Max DD ₹{fd:,.2f} • Profit Factor {"∞" if np.isinf(fp) else f"{fp:.2f}"}')
 fig=go.Figure();fig.add_trace(go.Scatter(x=oq.index.to_numpy(),y=oq.Equity.to_numpy(dtype=float),name='OOS Equity'));st.plotly_chart(fig,use_container_width=True);st.dataframe(ot,use_container_width=True)
 if oe>CAP and op>1 and len(ot)>=5:st.success('OOS PASS: profitable with PF > 1 and at least 5 trades.')
 else:st.error('OOS NOT PASSED. Keep real-money trading OFF.')
st.info('V8.1 keeps V8 regime/risk filters and makes OOS the primary validation. No real-money orders.')
