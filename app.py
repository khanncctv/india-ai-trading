import streamlit as st
import pandas as pd, numpy as np, yfinance as yf

st.set_page_config(page_title='India AI Trading V20', page_icon='🇮🇳', layout='wide')
CAPITAL=100000.0

@st.cache_data(ttl=300)
def load_data(symbol, period, interval):
    for s in [symbol, '^NSEI', 'NIFTYBEES.NS']:
        try:
            df=yf.download(s, period=period, interval=interval, auto_adjust=False, progress=False, threads=False)
            if isinstance(df.columns,pd.MultiIndex): df=df.droplevel(-1,axis=1)
            df.columns=[str(c) for c in df.columns]
            needed=['Open','High','Low','Close','Volume']
            if all(c in df.columns for c in needed) and len(df)>250:
                return df[needed].apply(pd.to_numeric,errors='coerce').dropna(),s
        except Exception:
            continue
    return pd.DataFrame(),'NONE'

def indicators(x):
    x=x.copy()
    x['EMA20']=x.Close.ewm(20,adjust=False).mean(); x['EMA50']=x.Close.ewm(50,adjust=False).mean(); x['EMA200']=x.Close.ewm(200,adjust=False).mean()
    x['MOM20']=x.Close.pct_change(20); x['MOM60']=x.Close.pct_change(60)
    x['VOL20']=x.Close.pct_change().rolling(20).std()*np.sqrt(252)
    x['VOL60']=x.Close.pct_change().rolling(60).std()*np.sqrt(252)
    x['VWAP']=(x.Close*x.Volume).cumsum()/x.Volume.cumsum(); x['VOLMA']=x.Volume.rolling(20).mean()
    d=x.Close.diff(); g=d.clip(lower=0).rolling(14).mean(); l=(-d.clip(upper=0)).rolling(14).mean(); x['RSI']=100-100/(1+g/l.replace(0,np.nan))
    tr=pd.concat([x.High-x.Low,(x.High-x.Close.shift()).abs(),(x.Low-x.Close.shift()).abs()],axis=1).max(axis=1); x['ATR']=tr.rolling(14).mean(); x['ATRpct']=x.ATR/x.Close
    return x.dropna()

def signal(r):
    trend=(r.Close/r.EMA200-1)*100
    # Conservative, non-optimized regime gate.
    return bool(trend>=1.0 and r.MOM20>=0.01 and r.MOM60>=0.03 and r.Close>r.EMA20>r.EMA50>r.EMA200 and r.Close>r.VWAP and r.Volume>=r.VOLMA and 50<=r.RSI<=68 and r.ATRpct<0.035)

def backtest(x, fee, slippage):
    cash=CAPITAL; pos=0; entry=0; trades=[]; equity=[]
    for ts,r in x.iterrows():
        p=float(r.Close); equity.append(cash+pos*p)
        exit_now=pos and (r.EMA20<r.EMA50 or r.Close<r.VWAP or r.MOM20<0 or r.MOM60<0)
        if exit_now:
            sell=p*(1-slippage); pnl=(sell-entry)-(sell+entry)*fee; cash+=sell-sell*fee; trades.append([ts,'SELL',sell,pnl]); pos=0
        if not pos and signal(r):
            buy=p*(1+slippage); cash-=buy+buy*fee; pos=1; entry=buy; trades.append([ts,'BUY',buy,0])
    if pos:
        p=float(x.Close.iloc[-1]); sell=p*(1-slippage); pnl=(sell-entry)-(sell+entry)*fee; cash+=sell-sell*fee; trades.append([x.index[-1],'SELL',sell,pnl])
    t=pd.DataFrame(trades,columns=['Time','Side','Price','PnL']); sells=t[t.Side=='SELL']
    wins=sells[sells.PnL>0].PnL; losses=sells[sells.PnL<0].PnL
    pf=float(wins.sum()/(-losses.sum())) if len(losses) else 0
    win=float((sells.PnL>0).mean()*100) if len(sells) else 0
    eq=pd.Series(equity+[cash]); peak=eq.cummax(); maxdd=float((peak-eq).max())
    daily=eq.pct_change().replace([np.inf,-np.inf],np.nan).dropna(); sharpe=float(daily.mean()/daily.std()*np.sqrt(252)) if daily.std()>0 else 0
    return cash,t,win,pf,maxdd,sharpe

def rolling_oos(x, fee, slip, windows=10):
    n=len(x); start=int(n*.35); span=max(30,(n-start)//windows); rows=[]
    for i in range(windows):
        a=start+i*span; b=min(n,a+span)
        z=x.iloc[a:b]
        if len(z)<30: continue
        end,t,w,pf,dd,sh=backtest(z,fee,slip)
        rows.append([i+1,end-CAPITAL,len(t[t.Side=='SELL']),w,pf,dd,sh])
    return pd.DataFrame(rows,columns=['Window','P&L','Trades','Win Rate','Profit Factor','Max DD','Sharpe'])

st.title('🇮🇳 India AI Trading Agent — V20 RESEARCH GRADE')
st.warning('PAPER TRADING ONLY — no real-money orders or broker API.')

sym=st.sidebar.text_input('Symbol','^NSEI'); period=st.sidebar.selectbox('Period',['3y','5y'],1); interval=st.sidebar.selectbox('Interval',['1d','1h'],0)
fee=st.sidebar.slider('Round-trip cost',0.0,0.004,0.001,0.0005); slip=st.sidebar.slider('Slippage',0.0,0.003,0.0005,0.0005)
raw,src=load_data(sym,period,interval)
if raw.empty: st.error('Market data unavailable. Try ^NSEI / 5y / 1d.'); st.stop()
df=indicators(raw); r=df.iloc[-1]; trend=float((r.Close/r.EMA200-1)*100); current=signal(r)
st.caption(f'Data source: {src} • Rows: {len(raw)} • Last bar: {raw.index[-1]}')
a,b,c,d,e=st.columns(5); a.metric('Last',f'{r.Close:,.2f}'); b.metric('RSI',f'{r.RSI:.1f}'); c.metric('Signal','BUY' if current else 'NO TRADE'); d.metric('Trend Edge',f'{trend:.2f}%'); e.metric('20D Momentum',f'{r.MOM20*100:.2f}%')

st.info('V20 deliberately refuses to trade unless trend, multi-horizon momentum, volume, volatility and price alignment agree. Research gates are locked against parameter chasing.')
if st.button('🚀 RUN V20 RESEARCH AUDIT',type='primary'):
    end,t,w,pf,dd,sh=backtest(df,fee,slip); strategy=end-CAPITAL
    bench=(float(df.Close.iloc[-1])/float(df.Close.iloc[0])-1)*CAPITAL
    wf=rolling_oos(df,fee,slip)
    prof=int((wf['P&L']>0).sum()) if len(wf) else 0; med=float(wf.P&L.median()) if len(wf) else 0; worst=float(wf.P&L.min()) if len(wf) else 0
    a,b,c,d,e,f=st.columns(6); a.metric('Strategy P&L',f'₹{strategy:,.2f}'); b.metric('Price Benchmark',f'₹{bench:,.2f}'); c.metric('Trades',len(t[t.Side=='SELL'])); d.metric('Win Rate',f'{w:.1f}%'); e.metric('Profit Factor',f'{pf:.2f}'); f.metric('Sharpe',f'{sh:.2f}')
    st.subheader('Rolling Out-of-Sample — Locked')
    st.dataframe(wf,use_container_width=True)
    a,b,c,d=st.columns(4); a.metric('Profitable OOS',f'{prof}/{len(wf)}'); b.metric('Median OOS',f'₹{med:,.2f}'); c.metric('Worst OOS',f'₹{worst:,.2f}'); d.metric('Max Drawdown',f'₹{dd:,.2f}')
    # Deliberately strict: strategy must show absolute and risk-adjusted evidence, not merely PF.
    gate=(len(wf)==10 and prof>=8 and med>0 and worst>-1500 and len(t[t.Side=='SELL'])>=20 and pf>=1.2 and sh>=0.5 and strategy>0)
    if gate: st.success('RESEARCH GATE PASS — still requires forward paper validation before any live consideration.')
    else: st.error('RESEARCH GATE FAIL — NO TRADE / PAPER ONLY. Do not enable real-money orders.')
    st.subheader('Trade History'); st.dataframe(t,use_container_width=True)

st.caption('V20 is a research framework, not a profit guarantee. A failed gate is a valid result.')
