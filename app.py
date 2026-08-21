import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier

st.set_page_config(page_title='India AI Trading Agent TEST', page_icon='🇮🇳', layout='wide')
CAPITAL=100000.0

@st.cache_data(ttl=300)
def get_data(symbol='^NSEI'):
    attempts=[(symbol,'5y','1d'),('^NSEI','5y','1d'),('NIFTYBEES.NS','5y','1d')]
    for s,p,i in attempts:
        try:
            x=yf.download(s,period=p,interval=i,auto_adjust=False,progress=False,threads=False)
            if isinstance(x.columns,pd.MultiIndex): x=x.droplevel(-1,axis=1)
            x.columns=[str(c) for c in x.columns]
            cols=['Open','High','Low','Close','Volume']
            if all(c in x.columns for c in cols):
                x=x[cols].apply(pd.to_numeric,errors='coerce').dropna()
                if len(x)>=250: return x,s
        except Exception:
            pass
    return pd.DataFrame(),''

def make_features(x):
    x=x.copy(); c=x.Close; r=c.pct_change()
    for n in [1,5,10,20,40,60]: x[f'ret{n}']=c.pct_change(n)
    for n in [20,50,200]: x[f'ema{n}']=c.ewm(n,adjust=False).mean()/c-1
    x['vol20']=r.rolling(20).std(); x['vol60']=r.rolling(60).std()
    x['volratio']=x.Volume/x.Volume.rolling(20).mean()
    d=c.diff(); up=d.clip(lower=0).rolling(14).mean(); dn=(-d.clip(upper=0)).rolling(14).mean(); x['rsi']=100-100/(1+up/dn.replace(0,np.nan))
    x['target']=c.shift(-5)/c-1
    return x.replace([np.inf,-np.inf],np.nan).dropna()

FEATURES=['ret1','ret5','ret10','ret20','ret40','ret60','ema20','ema50','ema200','vol20','vol60','volratio','rsi']

def train_predict(train,test):
    y=(train.target>0.005).astype(int)
    if y.nunique()<2: return np.full(len(test),.5)
    m=HistGradientBoostingClassifier(max_depth=3,max_iter=100,learning_rate=.05,min_samples_leaf=30,random_state=42,l2_regularization=1.0)
    m.fit(train[FEATURES],y)
    return m.predict_proba(test[FEATURES])[:,1]

st.title('🇮🇳 India AI Trading Agent — TEST RUN STABLE')
st.warning('PAPER TRADING ONLY — no real-money orders.')
st.write('This build is intentionally simple: first verify Test Run works, then add the research gates.')

raw,source=get_data('^NSEI')
if raw.empty:
    st.error('Market data unavailable. Yahoo/NIFTY data could not be loaded.')
    st.stop()

df=make_features(raw)
latest=df.iloc[[-1]].copy()
prob=float(train_predict(df.iloc[:-1],latest)[0])
signal='BUY' if prob>=0.62 else 'NO TRADE'

a,b,c,d=st.columns(4)
a.metric('Last',f"{latest.Close.iloc[0]:,.2f}")
b.metric('AI Probability',f'{prob*100:.1f}%')
c.metric('Signal',signal)
d.metric('Rows',len(raw))
st.caption(f'Data source: {source} • Last bar: {raw.index[-1]}')

if st.button('🧪 TEST RUN',type='primary'):
    try:
        split=int(len(df)*0.70)
        train=df.iloc[:split].copy(); test=df.iloc[split:].copy()
        p=train_predict(train,test)
        test=test.copy(); test['Probability']=p; test['Prediction']=np.where(p>=.62,1,0)
        test['Actual']=(test.target>0.005).astype(int)
        trades=test[test.Prediction==1]
        pnl=0.0
        for _,r in trades.iterrows():
            pnl += float(r.target)*CAPITAL*0.01
        win=(trades.Actual==1).mean()*100 if len(trades) else 0.0
        st.success('TEST RUN SUCCESSFUL ✅')
        a,b,c,d=st.columns(4)
        a.metric('OOS Signals',len(trades)); b.metric('Test P&L',f'₹{pnl:,.2f}'); c.metric('Signal Win Rate',f'{win:.1f}%'); d.metric('OOS Rows',len(test))
        st.subheader('Test OOS Results')
        st.dataframe(test[['Close','Probability','Prediction','Actual','target']].tail(100),use_container_width=True)
    except Exception as e:
        st.error('TEST RUN FAILED')
        st.exception(e)
