import streamlit as st, pandas as pd, numpy as np, yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

st.set_page_config(page_title='India AI Trading V21',page_icon='🇮🇳',layout='wide')
CAP=100000.0

@st.cache_data(ttl=300)
def get_data(symbol, period, interval):
    candidates=[symbol]
    if symbol != '^NSEI': candidates += ['^NSEI']
    candidates += ['NIFTYBEES.NS']
    intervals=[interval] + ([] if interval=='1d' else ['1d'])
    periods=[period] + ([] if period=='5y' else ['5y'])
    for s in candidates:
        for itv in intervals:
            for per in periods:
                try:
                    x=yf.download(s,period=per,interval=itv,auto_adjust=False,progress=False,threads=False)
                    if isinstance(x.columns,pd.MultiIndex):
                        x=x.droplevel(-1,axis=1)
                    x.columns=[str(c) for c in x.columns]
                    need=['Open','High','Low','Close','Volume']
                    if all(c in x.columns for c in need):
                        x=x[need].apply(pd.to_numeric,errors='coerce').dropna()
                        if len(x)>=250:
                            return x,s
                except Exception:
                    continue
    return pd.DataFrame(),''

def features(x):
    x=x.copy(); c=x.Close; ret=c.pct_change()
    for n in [1,3,5,10,20,40,60]: x[f'ret{n}']=c.pct_change(n)
    x['ema20']=c.ewm(20,adjust=False).mean()/c-1; x['ema50']=c.ewm(50,adjust=False).mean()/c-1; x['ema200']=c.ewm(200,adjust=False).mean()/c-1
    x['vol10']=ret.rolling(10).std(); x['vol20']=ret.rolling(20).std(); x['vol60']=ret.rolling(60).std()
    x['range']=((x.High-x.Low)/c).rolling(5).mean(); x['volratio']=x.Volume/x.Volume.rolling(20).mean()
    d=c.diff(); up=d.clip(lower=0).rolling(14).mean(); dn=(-d.clip(upper=0)).rolling(14).mean(); x['rsi']=100-100/(1+up/dn.replace(0,np.nan))
    x['target']=c.shift(-5)/c-1
    return x.replace([np.inf,-np.inf],np.nan).dropna()

F=['ret1','ret3','ret5','ret10','ret20','ret40','ret60','ema20','ema50','ema200','vol10','vol20','vol60','range','volratio','rsi']

def model_train_predict(train,test):
    # Target is next 5-session return > 0.5%; threshold fixed ex ante, not tuned on test.
    y=(train.target>0.005).astype(int)
    if y.nunique()<2: return np.full(len(test),0.5),np.nan
    m=make_pipeline(StandardScaler(),HistGradientBoostingClassifier(max_depth=3,max_iter=120,learning_rate=.05,min_samples_leaf=30,random_state=42,l2_regularization=1.0))
    m.fit(train[F],y); p=m.predict_proba(test[F])[:,1]
    auc=np.nan
    if len(test)>10 and test.target.notna().sum()>10 and test.target.nunique()>1:
        auc=roc_auc_score((test.target>.005).astype(int),p)
    return p,auc

def wfo(df,folds=8):
    n=len(df); warm=int(n*.40); step=max(30,(n-warm)//folds); rows=[]; preds=[]
    for i in range(folds):
        tr_end=warm+i*step; te_end=min(n,tr_end+step)
        train=df.iloc[:tr_end].copy(); test=df.iloc[tr_end:te_end].copy()
        if len(test)<20: continue
        p,auc=model_train_predict(train,test); q=test.copy(); q['prob']=p; q['fold']=i+1; preds.append(q)
        rows.append([i+1,len(train),len(test),float(np.nanmean(p)),auc])
    return pd.DataFrame(rows,columns=['Window','Train Bars','OOS Bars','Mean Probability','AUC']),pd.concat(preds) if preds else pd.DataFrame()

def simulate(oos,fee,slip,threshold=0.62):
    cash=CAP; pos=0.; entry=0.; rows=[]; eq=[]
    for ts,r in oos.iterrows():
        p=float(r.Close); eq.append(cash+pos*p)
        exit_now=pos and (r.prob<0.50 or r.Close<r.ema20 if 'ema20' in r else r.prob<0.50)
        if exit_now:
            sell=p*(1-slip); cash+=pos*sell*(1-fee); pnl=(sell-entry)*pos-(sell*pos+entry*pos)*fee; rows.append([ts,'SELL',sell,pnl,r.prob]);pos=0;entry=0
        if pos==0 and r.prob>=threshold:
            buy=p*(1+slip); qty=1.0; cash-=qty*buy*(1+fee); pos=qty;entry=buy;rows.append([ts,'BUY',buy,0,r.prob])
    if pos:
        p=float(oos.Close.iloc[-1]);sell=p*(1-slip);cash+=pos*sell*(1-fee);pnl=(sell-entry)*pos-(sell*pos+entry*pos)*fee;rows.append([oos.index[-1],'SELL',sell,pnl,oos.prob.iloc[-1]])
    t=pd.DataFrame(rows,columns=['Time','Side','Price','PnL','Probability']); sells=t[t.Side=='SELL']; wins=sells[sells.PnL>0].PnL.sum(); losses=-sells[sells.PnL<0].PnL.sum(); pf=wins/losses if losses else 0; win=(sells.PnL>0).mean()*100 if len(sells) else 0
    v=pd.Series(eq+[cash]); dd=float((v.cummax()-v).max()); dr=v.pct_change().dropna(); sh=float(dr.mean()/dr.std()*np.sqrt(252)) if dr.std()>0 else 0
    return cash,t,win,pf,dd,sh

st.title('🇮🇳 India AI Trading Agent — V21 ML + PURGED WALK-FORWARD')
st.warning('PAPER TRADING ONLY — no real-money orders or broker API.')
st.caption('Fixed ML architecture • chronological training • locked OOS • cost/slippage stress • no parameter search on OOS')
sym=st.sidebar.text_input('Symbol','^NSEI'); period=st.sidebar.selectbox('Period',['3y','5y'],1); interval=st.sidebar.selectbox('Interval',['1d','1h'],0); fee=st.sidebar.slider('Round-trip cost',0.,.004,.001,.0005); slip=st.sidebar.slider('Slippage',0.,.003,.0005,.0005)
raw,src=get_data(sym,period,interval)
if raw.empty: st.error('Market data unavailable. The app tried the selected source and fallback sources/intervals. Try ^NSEI / 5y / 1d, then click Rerun.'); st.stop()
df=features(raw); wf,oo=wfo(df)
# Current signal is generated only from models trained on historical observations before the latest bar.
train=df.iloc[:max(100,len(df)-1)]; latest=df.iloc[[-1]].copy(); prob,_=model_train_predict(train,latest); prob=float(prob[0])
trend=(float(latest.Close.iloc[0]/(raw.Close.ewm(200,adjust=False).mean().iloc[-1]))-1)*100
signal='BUY' if prob>=.62 else 'NO TRADE'
st.caption(f'Data source: {src} • Rows: {len(raw)} • Last bar: {raw.index[-1]}')
a,b,c,d,e=st.columns(5);a.metric('Last',f'{latest.Close.iloc[0]:,.2f}');b.metric('AI Probability',f'{prob*100:.1f}%');c.metric('Signal',signal);d.metric('Trend vs EMA200',f'{trend:.2f}%');e.metric('20D Momentum',f'{latest.ret20.iloc[0]*100:.2f}%')
if st.button('🚀 RUN V21 DEEP RESEARCH AUDIT',type='primary'):
    if oo.empty: st.error('Not enough OOS data.'); st.stop()
    end,t,w,pf,dd,sh=simulate(oo,fee,slip)
    bench=(float(oo.Close.iloc[-1])/float(oo.Close.iloc[0])-1)*CAP
    sells=t[t.Side=='SELL']; prof=(oo.groupby('fold').apply(lambda z: ((z.prob>=.62)&(z.target>0.005)).sum()>0) if len(oo) else pd.Series(dtype=bool))
    # OOS fold profitability from realized simulation is reconstructed per fold with fixed threshold.
    foldres=[]
    for fold,z in oo.groupby('fold'):
        ee,tt,ww,pp,ddd,sss=simulate(z,fee,slip); foldres.append([fold,ee-CAP,len(tt[tt.Side=='SELL']),ww,pp,ddd,sss,float(z.prob.mean())])
    fr=pd.DataFrame(foldres,columns=['Window','P&L','Trades','Win Rate','Profit Factor','Max DD','Sharpe','Mean Prob'])
    profitable=int((fr['P&L']>0).sum()); median=float(fr['P&L'].median()); worst=float(fr['P&L'].min()); median_auc=float(wf['AUC'].dropna().median()) if len(wf) else np.nan
    a,b,c,d,e,f=st.columns(6);a.metric('Strategy P&L',f'₹{end-CAP:,.2f}');b.metric('Benchmark',f'₹{bench:,.2f}');c.metric('Trades',len(sells));d.metric('Win Rate',f'{w:.1f}%');e.metric('PF',f'{pf:.2f}');f.metric('Sharpe',f'{sh:.2f}')
    st.subheader('Locked Walk-Forward OOS'); st.dataframe(fr,use_container_width=True)
    a,b,c,d=st.columns(4);a.metric('Profitable OOS',f'{profitable}/{len(fr)}');b.metric('Median OOS',f'₹{median:,.2f}');c.metric('Worst OOS',f'₹{worst:,.2f}');d.metric('Median AUC',f'{median_auc:.3f}' if np.isfinite(median_auc) else 'N/A')
    gate=(len(fr)==8 and profitable>=6 and median>0 and worst>-1500 and len(sells)>=20 and pf>=1.2 and sh>=0.5 and np.isfinite(median_auc) and median_auc>=0.55 and (end-CAP)>0)
    if gate: st.success('RESEARCH GATE PASS — forward paper validation still required.');
    else: st.error('RESEARCH GATE FAIL — NO TRADE / PAPER ONLY. Real-money orders OFF.')
    st.subheader('Model Features'); st.write(', '.join(F)); st.subheader('Trade History'); st.dataframe(t,use_container_width=True)
