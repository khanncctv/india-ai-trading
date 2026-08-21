import os, numpy as np, pandas as pd, streamlit as st, plotly.graph_objects as go, yfinance as yf
st.set_page_config(page_title='India AI Trading V7',page_icon='🇮🇳',layout='wide')
CAP=float(os.getenv('STARTING_CAPITAL','100000')); Q=int(os.getenv('TRADE_QTY','1')); SLIP=.0005; FEE=.0005
@st.cache_data(ttl=300)
def load_data(symbol,period,interval):
    x=yf.download(symbol,period=period,interval=interval,auto_adjust=False,progress=False)
    if isinstance(x.columns,pd.MultiIndex):
        if symbol in x.columns.get_level_values(1): x=x.xs(symbol,axis=1,level=1,drop_level=True)
        else: x=x.droplevel(1,axis=1)
    x.columns=[str(c) for c in x.columns]
    for c in ['Open','High','Low','Close','Volume']:
        if c in x: x[c]=pd.to_numeric(x[c],errors='coerce')
    return x.dropna()
def features(x):
    x=x.copy(); x['EMA20']=x.Close.ewm(span=20,adjust=False).mean(); x['EMA50']=x.Close.ewm(span=50,adjust=False).mean()
    tr=pd.concat([x.High-x.Low,(x.High-x.Close.shift()).abs(),(x.Low-x.Close.shift()).abs()],axis=1).max(axis=1); x['ATR']=tr.rolling(14).mean()
    d=x.Close.diff(); g=d.clip(lower=0).rolling(14).mean(); l=(-d.clip(upper=0)).rolling(14).mean(); x['RSI']=100-100/(1+g/l.replace(0,np.nan))
    x['VWAP']=(x.Close*x.Volume).cumsum()/x.Volume.cumsum(); x['VOLMA']=x.Volume.rolling(20).mean(); return x.dropna()
def backtest(x,capital=CAP):
    cash=capital; pos=0; entry=stop=target=0; peak=capital; maxdd=0; streak=0; rows=[]; equity=[]
    for ts,r in x.iterrows():
        p=float(r.Close); eq=cash+pos*p; peak=max(peak,eq); maxdd=max(maxdd,peak-eq); equity.append([ts,eq])
        if pos:
            reason='STOP' if p<=stop else 'TARGET' if p>=target else 'EXIT' if r.EMA20<r.EMA50 or p<r.VWAP else None
            if reason:
                ep=p*(1-SLIP); pnl=(ep-entry)*pos-((ep+entry)*pos*FEE); cash+=ep*pos-ep*pos*FEE; rows.append([ts,'SELL',ep,pnl,reason]); streak=streak+1 if pnl<0 else 0; pos=0
        elif streak<3 and r.EMA20>r.EMA50 and 52<=r.RSI<=65 and r.Close>r.VWAP and r.Volume>=r.VOLMA:
            risk=1.2*float(r.ATR); ep=p*(1+SLIP)
            if risk>0 and ep*Q<=cash: cash-=ep*Q; pos=Q; entry=ep; stop=ep-risk; target=ep+2*risk; rows.append([ts,'BUY',ep,0,'ENTRY'])
    if pos:
        ep=float(x.iloc[-1].Close)*(1-SLIP); pnl=(ep-entry)*pos-((ep+entry)*pos*FEE); cash+=ep*pos-ep*pos*FEE; rows.append([x.index[-1],'SELL',ep,pnl,'END'])
    t=pd.DataFrame(rows,columns=['Time','Side','Price','PnL','Reason']); s=t[t.Side=='SELL']; gp=s.loc[s.PnL>0,'PnL'].sum(); gl=-s.loc[s.PnL<0,'PnL'].sum(); pf=gp/gl if gl else (np.inf if gp>0 else 0); win=(s.PnL>0).mean()*100 if len(s) else 0
    eq=pd.DataFrame(equity,columns=['Time','Equity']).set_index('Time'); return cash,t,win,maxdd,pf,eq
def walk_forward(df,windows=4):
    n=len(df)
    if n<300:return pd.DataFrame()
    first=max(50,n//5); step=max(20,(n-first)//windows); out=[]
    for i in range(windows):
        start=first+i*step; end=min(n,start+step); test=df.iloc[start:end]
        if len(test)<20:continue
        ending,trades,win,dd,pf,_=backtest(test,CAP); out.append([i+1,test.index[0],test.index[-1],ending-CAP,len(trades),win,dd,pf])
    return pd.DataFrame(out,columns=['Window','Start','End','P&L','Trades','Win Rate','Max DD','Profit Factor'])
st.title('🇮🇳 India AI Trading Agent — V7 WALK-FORWARD ROBUST'); st.warning('PAPER TRADING ONLY — no real-money orders.')
symbol=st.sidebar.text_input('Symbol','^NSEI'); period=st.sidebar.selectbox('Period',['2y','3y','5y'],2); interval=st.sidebar.selectbox('Interval',['1d','1h'],0)
raw=load_data(symbol,period,interval)
if raw.empty or not {'High','Low','Close','Volume'}.issubset(raw.columns): st.error('Market data unavailable. Try ^NSEI with 1d.'); st.stop()
df=features(raw); r=df.iloc[-1]; signal=bool(r.EMA20>r.EMA50 and 52<=r.RSI<=65 and r.Close>r.VWAP and r.Volume>=r.VOLMA)
a,b,c,d=st.columns(4); a.metric('Last',f'{float(r.Close):,.2f}'); b.metric('RSI',f'{float(r.RSI):.1f}'); c.metric('Signal','BUY' if signal else 'HOLD'); d.metric('ATR',f'{float(r.ATR):.2f}')
if st.button('🚀 RUN V7 WALK-FORWARD TEST',type='primary'):
    end,trades,win,dd,pf,eq=backtest(df); wf=walk_forward(df)
    st.subheader('Full Period'); a,b,c,d,e=st.columns(5); a.metric('Ending Value',f'₹{end:,.2f}'); b.metric('P&L',f'₹{end-CAP:,.2f}'); c.metric('Trades',len(trades)); d.metric('Win Rate',f'{win:.1f}%'); e.metric('Max DD',f'₹{dd:,.2f}'); st.metric('Profit Factor','∞' if np.isinf(pf) else f'{pf:.2f}')
    st.subheader('Walk-Forward Windows')
    if len(wf):
        st.dataframe(wf,use_container_width=True); profitable=int((wf['P&L']>0).sum()); total=len(wf); avg=float(wf['P&L'].mean()); a,b=st.columns(2); a.metric('Profitable Windows',f'{profitable}/{total}'); b.metric('Average Window P&L',f'₹{avg:,.2f}');
        if profitable>=((total+1)//2) and avg>0: st.success('Walk-forward result is encouraging.')
        else: st.error('Walk-forward result is not consistently profitable. Keep real-money trading OFF.')
    else: st.warning('Not enough data for walk-forward validation.')
    st.subheader('Equity Curve'); fig=go.Figure(); fig.add_trace(go.Scatter(x=eq.index.to_numpy(),y=eq.Equity.to_numpy(dtype=float),name='Equity')); st.plotly_chart(fig,use_container_width=True)
    st.subheader('Trade History'); st.dataframe(trades,use_container_width=True)
st.info('V7 uses multiple sequential test windows plus fees, slippage, ATR risk, 2R target and a 3-loss halt. Validation only; future returns are not guaranteed.')
