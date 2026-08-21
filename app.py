import streamlit as st, pandas as pd, numpy as np, yfinance as yf
st.set_page_config(page_title='India AI Trading V16',page_icon='🇮🇳',layout='wide')
CAP=100000
@st.cache_data(ttl=300)
def fetch(sym,period,interval):
    for s in [sym,'^NSEI','NIFTYBEES.NS']:
        try:
            x=yf.download(s,period=period,interval=interval,auto_adjust=False,progress=False,threads=False)
            if isinstance(x.columns,pd.MultiIndex): x=x.droplevel(-1,axis=1)
            x.columns=[str(c) for c in x.columns]
            if all(c in x for c in ['Open','High','Low','Close','Volume']) and len(x)>250:
                return x.apply(pd.to_numeric,errors='coerce').dropna(),s
        except Exception: pass
    return pd.DataFrame(),''
def features(x):
    x=x.copy(); x['E20']=x.Close.ewm(span=20,adjust=False).mean(); x['E50']=x.Close.ewm(span=50,adjust=False).mean(); x['E200']=x.Close.ewm(span=200,adjust=False).mean()
    tr=pd.concat([x.High-x.Low,(x.High-x.Close.shift()).abs(),(x.Low-x.Close.shift()).abs()],axis=1).max(axis=1); x['ATR']=tr.rolling(14).mean(); d=x.Close.diff(); g=d.clip(lower=0).rolling(14).mean(); l=(-d.clip(upper=0)).rolling(14).mean(); x['RSI']=100-100/(1+g/l.replace(0,np.nan)); x['VWAP']=(x.Close*x.Volume).cumsum()/x.Volume.cumsum(); x['VM']=x.Volume.rolling(20).mean(); x['EDGE']=(x.Close/x.E200-1)*100; return x.dropna()
def run(x,fee=.001,rr=2):
 cash=CAP; pos=0; entry=stop=target=0; rows=[]; peak=CAP; dd=0
 for ts,r in x.iterrows():
  p=float(r.Close); eq=cash+pos*p; peak=max(peak,eq); dd=max(dd,peak-eq)
  if pos and (p<=stop or p>=target or r.E20<r.E50):
   pnl=(p-entry)-(p+entry)*fee; cash+=p-p*fee; rows.append([ts,'SELL',p,pnl]); pos=0
  if not pos and r.E20>r.E50 and r.Close>r.E200 and r.Close>r.VWAP and r.Volume>=r.VM and 50<=r.RSI<=68 and r.EDGE>=1.0 and r.ATR/r.Close<.012:
   a=float(r.ATR); cash-=p; pos=1; entry=p; stop=p-a; target=p+rr*a; rows.append([ts,'BUY',p,0])
 if pos:
  p=float(x.Close.iloc[-1]); pnl=(p-entry)-(p+entry)*fee; cash+=p-p*fee; rows.append([x.index[-1],'SELL',p,pnl])
 t=pd.DataFrame(rows,columns=['Time','Side','Price','PnL']); s=t[t.Side=='SELL']; gp=s.loc[s.PnL>0,'PnL'].sum(); gl=-s.loc[s.PnL<0,'PnL'].sum(); pf=gp/gl if gl else 0; win=(s.PnL>0).mean()*100 if len(s) else 0; return cash,t,win,dd,pf
def oos(x,n=10):
 start=int(len(x)*.2); span=max(30,(len(x)-start)//n); out=[]
 for i in range(n):
  z=x.iloc[start+i*span:min(len(x),start+(i+1)*span)]
  if len(z)>=30:
   e,t,w,d,p=run(z); out.append([i+1,z.index[0],z.index[-1],e-CAP,len(t),w,d,p])
 return pd.DataFrame(out,columns=['Window','Start','End','P&L','Trades','Win Rate','Max DD','PF'])
st.title('🇮🇳 India AI Trading Agent — V16 TRADE QUALITY + OOS'); st.warning('PAPER TRADING ONLY — no real-money orders.')
sym=st.sidebar.text_input('Symbol','^NSEI'); period=st.sidebar.selectbox('Period',['3y','5y'],1); interval=st.sidebar.selectbox('Interval',['1d','1h'],0); fee=st.sidebar.slider('Cost / round-trip',0.0,.003,.001,.0005)
raw,src=fetch(sym,period,interval)
if raw.empty: st.error('Market data unavailable. Try 1d and ^NSEI.'); st.stop()
df=features(raw); r=df.iloc[-1]; signal=r.EDGE>=1 and r.E20>r.E50 and r.Close>r.E200 and r.Close>r.VWAP and r.Volume>=r.VM and 50<=r.RSI<=68 and r.ATR/r.Close<.012
st.caption(f'Data source: {src} • Rows: {len(raw)} • Last bar: {raw.index[-1]}')
a,b,c,d=st.columns(4); a.metric('Last',f'{r.Close:,.2f}'); b.metric('RSI',f'{r.RSI:.1f}'); c.metric('Signal','BUY' if signal else 'HOLD'); d.metric('Trend Edge',f'{r.EDGE:.2f}%')
if st.button('RUN V16 FULL OOS',type='primary'):
 end,t,w,dd,pf=run(df,fee); wf=oos(df); bench=(float(df.Close.iloc[-1])/float(df.Close.iloc[0])-1)*CAP
 a,b,c,d,e=st.columns(5); a.metric('Strategy P&L',f'₹{end-CAP:,.2f}'); b.metric('Benchmark P&L',f'₹{bench:,.2f}'); c.metric('Trades',len(t)); d.metric('Win Rate',f'{w:.1f}%'); e.metric('Profit Factor',f'{pf:.2f}')
 st.subheader('10-Window OOS'); st.dataframe(wf,use_container_width=True)
 if len(wf):
  prof=int((wf['P&L']>0).sum()); med=float(wf['P&L'].median()); avg=float(wf['P&L'].mean()); worst=float(wf['P&L'].min()); a,b,c,d=st.columns(4); a.metric('Profitable Windows',f'{prof}/{len(wf)}'); b.metric('Median OOS',f'₹{med:,.2f}'); c.metric('Average OOS',f'₹{avg:,.2f}'); d.metric('Worst OOS',f'₹{worst:,.2f}')
  passed=prof>=8 and med>0 and avg>0 and worst>-1500 and pf>=1 and end-CAP>bench and int((wf['Trades']>=5).sum())>=8
  st.success('RESEARCH PASS — continue forward paper validation.') if passed else st.error('PAPER-ONLY — gate failed; real-money orders stay OFF.')
 st.subheader('Trade History'); st.dataframe(t,use_container_width=True)
