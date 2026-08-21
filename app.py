import streamlit as st, pandas as pd, numpy as np, yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

st.set_page_config(page_title='India AI Trading Agent V100',page_icon='🇮🇳',layout='wide')
CAP=100000.0

@st.cache_data(ttl=300)
def get_data(symbol='^NSEI'):
    for s in [symbol,'^NSEI','NIFTYBEES.NS']:
        try:
            x=yf.download(s,period='5y',interval='1d',auto_adjust=False,progress=False,threads=False)
            if isinstance(x.columns,pd.MultiIndex): x=x.droplevel(-1,axis=1)
            x.columns=[str(c) for c in x.columns]
            need=['Open','High','Low','Close','Volume']
            if all(c in x.columns for c in need):
                x=x[need].apply(pd.to_numeric,errors='coerce').dropna()
                if len(x)>=500:return x,s
        except Exception: pass
    return pd.DataFrame(),''

def make_features(x):
    x=x.copy(); c=x.Close; r=c.pct_change();
    for n in [1,2,3,5,10,20,40,60,120]: x[f'ret{n}']=c.pct_change(n)
    for n in [10,20,50,100,200]: x[f'ema{n}']=c.ewm(n,adjust=False).mean()/c-1
    x['range']=(x.High-x.Low)/c; x['gap']=x.Open/x.Close.shift(1)-1
    x['vol10']=r.rolling(10).std(); x['vol20']=r.rolling(20).std(); x['vol60']=r.rolling(60).std()
    x['volratio']=x.Volume/x.Volume.rolling(20).mean(); x['atrpct']=(x.High-x.Low).rolling(14).mean()/c
    d=c.diff(); up=d.clip(lower=0).rolling(14).mean(); dn=(-d.clip(upper=0)).rolling(14).mean(); x['rsi']=100-100/(1+up/dn.replace(0,np.nan))
    x['trend']=c/c.rolling(200).mean()-1
    # Fixed, pre-declared target: positive 5-session forward return above 0.5%.
    x['target']=((c.shift(-5)/c-1)>0.005).astype(float)
    x.iloc[-5:,x.columns.get_loc('target')]=np.nan
    return x.replace([np.inf,-np.inf],np.nan).dropna()

FEATURES=[c for c in make_features(pd.DataFrame({'Open':pd.Series(dtype=float),'High':pd.Series(dtype=float),'Low':pd.Series(dtype=float),'Close':pd.Series(dtype=float),'Volume':pd.Series(dtype=float)})).columns if c not in ['Open','High','Low','Close','Volume','target']] if False else ['ret1','ret2','ret3','ret5','ret10','ret20','ret40','ret60','ret120','ema10','ema20','ema50','ema100','ema200','range','gap','vol10','vol20','vol60','volratio','atrpct','rsi','trend']

def train_predict(train,test):
    y=train.target.astype(int)
    if y.nunique()<2:return np.full(len(test),.5)
    base=HistGradientBoostingClassifier(max_depth=3,max_iter=160,learning_rate=.04,min_samples_leaf=35,l2_regularization=1.5,random_state=42)
    model=CalibratedClassifierCV(base,method='sigmoid',cv=3)
    model.fit(train[FEATURES],y)
    return model.predict_proba(test[FEATURES])[:,1]

def purged_wfo(df,folds=10,embargo=5):
    n=len(df); first=int(n*.40); step=max(30,(n-first)//folds); pieces=[]; stats=[]
    for i in range(folds):
        train_end=first+i*step; test_start=min(n,train_end+embargo); test_end=min(n,test_start+step)
        tr=df.iloc[:train_end].dropna(subset=['target']); te=df.iloc[test_start:test_end].copy()
        if len(tr)<250 or len(te)<20:continue
        p=train_predict(tr,te); te['prob']=p;te['fold']=i+1;pieces.append(te)
        auc=np.nan
        if te.target.nunique()>1:auc=roc_auc_score(te.target.astype(int),p)
        stats.append([i+1,len(tr),len(te),float(np.mean(p)),auc])
    return pd.DataFrame(stats,columns=['Window','Train','OOS','Mean Prob','AUC']),pd.concat(pieces) if pieces else pd.DataFrame()

def simulate(z,fee,slip,threshold=.62):
    cash=CAP; pos=False; entry=0; rows=[]; eq=[]
    for ts,r in z.iterrows():
        p=float(r.Close);eq.append(cash+(p if pos else 0))
        if pos and (r.prob<.50 or r.Close<r.Close/(1+r.ema20)):
            sell=p*(1-slip); pnl=(sell-entry)-(sell+entry)*fee;cash+=sell-sell*fee;rows.append([ts,'SELL',sell,pnl,r.prob]);pos=False
        if not pos and r.prob>=threshold:
            buy=p*(1+slip);cash-=buy+buy*fee;entry=buy;pos=True;rows.append([ts,'BUY',buy,0,r.prob])
    if pos:
        p=float(z.Close.iloc[-1]);sell=p*(1-slip);pnl=(sell-entry)-(sell+entry)*fee;cash+=sell-sell*fee;rows.append([z.index[-1],'SELL',sell,pnl,z.prob.iloc[-1]])
    t=pd.DataFrame(rows,columns=['Time','Side','Price','PnL','Probability']);s=t[t.Side=='SELL'];wins=s.loc[s.PnL>0,'PnL'].sum();loss=-s.loc[s.PnL<0,'PnL'].sum();pf=wins/loss if loss else 0;wr=(s.PnL>0).mean()*100 if len(s) else 0;v=pd.Series(eq+[cash]);dd=float((v.cummax()-v).max());q=v.pct_change().dropna();sh=float(q.mean()/q.std()*np.sqrt(252)) if q.std()>0 else 0
    return cash,t,wr,pf,dd,sh

st.title('🇮🇳 India AI Trading Agent — V100 FINAL RESEARCH FRAMEWORK')
st.warning('PAPER TRADING ONLY — no real-money orders or broker API.')
st.caption('Fixed ML • calibrated probability • purged/embargo walk-forward • locked OOS • benchmark • cost/slippage stress • strict gate • no OOS parameter search')
raw,src=get_data(st.sidebar.text_input('Symbol','^NSEI'))
fee=st.sidebar.slider('Round-trip cost',0.,.004,.001,.0005);slip=st.sidebar.slider('Slippage',0.,.003,.0005,.0005)
if raw.empty:st.error('Market data unavailable. Try ^NSEI and rerun.');st.stop()
df=make_features(raw);train=df.iloc[:-1];latest=df.iloc[[-1]];prob=float(train_predict(train,latest)[0]);trend=float(latest.trend.iloc[0]*100);signal='BUY' if prob>=.62 else 'NO TRADE'
st.caption(f'Data source: {src} • Rows: {len(raw)} • Last bar: {raw.index[-1]}')
a,b,c,d,e=st.columns(5);a.metric('Last',f'{latest.Close.iloc[0]:,.2f}');b.metric('AI Probability',f'{prob*100:.1f}%');c.metric('Signal',signal);d.metric('Trend',f'{trend:.2f}%');e.metric('20D Momentum',f'{latest.ret20.iloc[0]*100:.2f}%')
if st.button('🚀 RUN V100 DEEP AUDIT',type='primary'):
    wf,oo=purged_wfo(df)
    if oo.empty:st.error('Not enough data for locked OOS.');st.stop()
    end,t,wr,pf,dd,sh=simulate(oo,fee,slip);sells=t[t.Side=='SELL']; bench=(float(oo.Close.iloc[-1])/float(oo.Close.iloc[0])-1)*CAP
    rows=[]
    for k,z in oo.groupby('fold'):
        ee,tt,ww,pp,ddd,sss=simulate(z,fee,slip);rows.append([k,ee-CAP,len(tt[tt.Side=='SELL']),ww,pp,ddd,sss])
    fr=pd.DataFrame(rows,columns=['Window','P&L','Trades','Win Rate','PF','Max DD','Sharpe']); prof=int((fr['P&L']>0).sum()); med=float(fr['P&L'].median()); worst=float(fr['P&L'].min()); auc=float(wf.AUC.dropna().median()) if len(wf.AUC.dropna()) else np.nan
    a,b,c,d,e,f=st.columns(6);a.metric('Strategy P&L',f'₹{end-CAP:,.2f}');b.metric('Benchmark',f'₹{bench:,.2f}');c.metric('Trades',len(sells));d.metric('Win Rate',f'{wr:.1f}%');e.metric('PF',f'{pf:.2f}');f.metric('Sharpe',f'{sh:.2f}')
    st.subheader('10-Window Purged OOS');st.dataframe(fr,use_container_width=True)
    a,b,c,d=st.columns(4);a.metric('Profitable OOS',f'{prof}/{len(fr)}');b.metric('Median OOS',f'₹{med:,.2f}');c.metric('Worst OOS',f'₹{worst:,.2f}');d.metric('Median AUC',f'{auc:.3f}' if np.isfinite(auc) else 'N/A')
    gate=(len(fr)==10 and prof>=8 and med>0 and worst>-1500 and len(sells)>=20 and pf>=1.2 and sh>=.5 and np.isfinite(auc) and auc>=.55 and end-CAP>0 and end-CAP>bench*.15)
    if gate:st.success('V100 RESEARCH GATE PASS — forward paper validation required; live trading remains OFF.')
    else:st.error('V100 RESEARCH GATE FAIL — NO TRADE. Real-money orders OFF.')
    st.subheader('Model / Validation');st.write('Features:',FEATURES);st.write('Purged embargo:',5,'bars • OOS folds:',len(fr));st.subheader('Trade History');st.dataframe(t,use_container_width=True)
