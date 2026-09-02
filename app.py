import os
import math
import requests
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder='.', static_url_path='')
API_KEY = os.getenv('TWELVE_DATA_API_KEY', '').strip()
NEWS_API_KEY = os.getenv('ALPHAVANTAGE_API_KEY', '').strip()
BASE = 'https://api.twelvedata.com'
AV_BASE = 'https://www.alphavantage.co/query'
STRATEGIES = ['ema','rsi','macd','bollinger','breakout']


def td_get(path, params):
    if not API_KEY:
        raise RuntimeError('TWELVE_DATA_API_KEY ontbreekt op de server.')
    params = dict(params)
    params['apikey'] = API_KEY
    r = requests.get(BASE + path, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and data.get('status') == 'error':
        raise RuntimeError(data.get('message', 'Twelve Data fout'))
    return data


def news_for(symbol):
    if not NEWS_API_KEY:
        return {'available': False, 'score': 50, 'label': 'Niet gekoppeld', 'items': [], 'message': 'Voeg ALPHAVANTAGE_API_KEY toe voor nieuws en sentiment.'}
    try:
        r = requests.get(AV_BASE, params={'function':'NEWS_SENTIMENT','tickers':symbol,'limit':20,'sort':'LATEST','apikey':NEWS_API_KEY}, timeout=20)
        r.raise_for_status()
        data = r.json()
        feed = data.get('feed', [])[:12]
        items=[]; scores=[]
        for item in feed:
            ticker_score = None
            for ts in item.get('ticker_sentiment', []):
                if str(ts.get('ticker','')).upper() == symbol.upper():
                    try: ticker_score=float(ts.get('ticker_sentiment_score'))
                    except Exception: ticker_score=None
                    break
            if ticker_score is None:
                try: ticker_score=float(item.get('overall_sentiment_score',0))
                except Exception: ticker_score=0
            scores.append(ticker_score)
            items.append({'title':item.get('title',''), 'source':item.get('source',''), 'url':item.get('url',''), 'time_published':item.get('time_published',''), 'sentiment':round(ticker_score,3)})
        avg=float(np.mean(scores)) if scores else 0
        score=max(0,min(100,50 + avg*50))
        label='Positief' if score>=60 else ('Negatief' if score<=40 else 'Neutraal')
        return {'available': True, 'score': round(score,1), 'label': label, 'items': items, 'message': ''}
    except Exception as e:
        return {'available': False, 'score': 50, 'label': 'Onbeschikbaar', 'items': [], 'message': str(e)}


def to_df(values):
    df = pd.DataFrame(values)
    if df.empty: return df
    for c in ['open','high','low','close','volume']:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df.sort_values('datetime').reset_index(drop=True)

def ema(s,n): return s.ewm(span=n,adjust=False).mean()
def rsi(s,n=14):
    d=s.diff(); up=d.clip(lower=0).rolling(n).mean(); down=(-d.clip(upper=0)).rolling(n).mean(); rs=up/down.replace(0,np.nan); return 100-(100/(1+rs))

def enrich(df):
    x=df.copy(); x['ema20']=ema(x.close,20); x['ema50']=ema(x.close,50); x['ema12']=ema(x.close,12); x['ema26']=ema(x.close,26); x['macd']=x.ema12-x.ema26; x['macd_signal']=ema(x.macd,9); x['rsi']=rsi(x.close,14)
    mid=x.close.rolling(20).mean(); sd=x.close.rolling(20).std(); x['bb_mid']=mid; x['bb_upper']=mid+2*sd; x['bb_lower']=mid-2*sd; x['breakout_high']=x.high.shift(1).rolling(20).max(); x['breakout_low']=x.low.shift(1).rolling(10).min()
    return x

def signals(df,strategy):
    x=enrich(df); x['buy']=False; x['sell']=False
    if strategy=='ema': x['buy']=(x.ema20>x.ema50)&(x.ema20.shift(1)<=x.ema50.shift(1)); x['sell']=(x.ema20<x.ema50)&(x.ema20.shift(1)>=x.ema50.shift(1))
    elif strategy=='rsi': x['buy']=(x.rsi>30)&(x.rsi.shift(1)<=30); x['sell']=(x.rsi<70)&(x.rsi.shift(1)>=70)
    elif strategy=='macd': x['buy']=(x.macd>x.macd_signal)&(x.macd.shift(1)<=x.macd_signal.shift(1)); x['sell']=(x.macd<x.macd_signal)&(x.macd.shift(1)>=x.macd_signal.shift(1))
    elif strategy=='bollinger': x['buy']=(x.close>x.bb_lower)&(x.close.shift(1)<=x.bb_lower.shift(1)); x['sell']=(x.close<x.bb_mid)&(x.close.shift(1)>=x.bb_mid.shift(1))
    elif strategy=='breakout': x['buy']=x.close>x.breakout_high; x['sell']=x.close<x.breakout_low
    return x

def backtest(df,strategy,fee_pct=0.10):
    x=signals(df,strategy); in_pos=False; entry=0.0; entry_time=None; trades=[]
    for _,row in x.iterrows():
        if not in_pos and bool(row.buy) and pd.notna(row.close): in_pos=True; entry=float(row.close); entry_time=row.datetime
        elif in_pos and bool(row.sell) and pd.notna(row.close):
            exitp=float(row.close); net=((exitp-entry)/entry*100)-(2*fee_pct); trades.append({'entry_time':str(entry_time),'exit_time':str(row.datetime),'entry':entry,'exit':exitp,'return_pct':net}); in_pos=False
    if in_pos and len(x):
        exitp=float(x.iloc[-1].close); net=((exitp-entry)/entry*100)-(2*fee_pct); trades.append({'entry_time':str(entry_time),'exit_time':str(x.iloc[-1].datetime),'entry':entry,'exit':exitp,'return_pct':net})
    returns=[t['return_pct'] for t in trades]; wins=[r for r in returns if r>0]; losses=[r for r in returns if r<=0]; equity=1.0; peak=1.0; max_dd=0.0
    for r in returns: equity*=1+r/100; peak=max(peak,equity); max_dd=min(max_dd,(equity/peak-1)*100)
    gp=sum(wins); gl=abs(sum(losses)); pf=gp/gl if gl>0 else (999 if gp>0 else 0)
    return {'trades':len(trades),'winrate':round(len(wins)/len(trades)*100,1) if trades else 0,'total_return':round((equity-1)*100,2),'avg_trade':round(np.mean(returns),2) if returns else 0,'profit_factor':round(pf,2) if math.isfinite(pf) else 999,'max_drawdown':round(max_dd,2),'trade_list':trades[-20:]}

def strategy_rank(bt):
    trades=min(bt['trades'],30)/30
    pf=min(bt['profit_factor'],3)/3
    ret=max(-1,min(1,bt['total_return']/25))
    dd=max(0,1-min(abs(bt['max_drawdown']),25)/25)
    return round(100*(0.35*((ret+1)/2)+0.25*pf+0.20*dd+0.20*trades),1)

def technical_score(df):
    x=enrich(df); r=x.iloc[-1]; score=50; reasons=[]
    if pd.notna(r.ema20) and pd.notna(r.ema50):
        if r.ema20>r.ema50: score+=15; reasons.append('Korte trend ligt boven lange trend')
        else: score-=15; reasons.append('Korte trend ligt onder lange trend')
    if pd.notna(r.rsi):
        if 45<=r.rsi<=65: score+=10; reasons.append('RSI ligt in een gezonde positieve zone')
        elif r.rsi>75: score-=10; reasons.append('RSI is sterk overbought')
        elif r.rsi<30: score-=5; reasons.append('RSI is oversold')
    if pd.notna(r.macd) and pd.notna(r.macd_signal):
        if r.macd>r.macd_signal: score+=10; reasons.append('MACD is positief')
        else: score-=10; reasons.append('MACD is negatief')
    if pd.notna(r.breakout_high) and r.close>r.breakout_high: score+=10; reasons.append('Koers breekt boven recente weerstand')
    return max(0,min(100,score)),reasons

@app.get('/')
def home(): return send_from_directory('.','index.html')
@app.get('/api/search')
def search():
    q=request.args.get('q','').strip()
    if not q:return jsonify([])
    try:return jsonify(td_get('/symbol_search',{'symbol':q,'outputsize':8}).get('data',[]))
    except Exception as e:return jsonify(error=str(e)),400

@app.get('/api/analyze')
def analyze():
    symbol=request.args.get('symbol','').strip().upper(); interval=request.args.get('interval','1day'); strategy=request.args.get('strategy','ema'); outputsize=min(int(request.args.get('outputsize','500')),5000)
    if not symbol:return jsonify(error='Geen ticker opgegeven.'),400
    try:
        series=td_get('/time_series',{'symbol':symbol,'interval':interval,'outputsize':outputsize,'order':'ASC'}); quote=td_get('/quote',{'symbol':symbol}); df=to_df(series.get('values',[]))
        if df.empty:raise RuntimeError('Geen koersdata gevonden.')
        x=signals(df,strategy); bt=backtest(df,strategy); points=[]
        for _,r in x.iterrows(): points.append({'time':r.datetime.isoformat(),'open':None if pd.isna(r.open) else round(float(r.open),6),'high':None if pd.isna(r.high) else round(float(r.high),6),'low':None if pd.isna(r.low) else round(float(r.low),6),'close':None if pd.isna(r.close) else round(float(r.close),6),'volume':None if 'volume' not in x.columns or pd.isna(r.volume) else float(r.volume),'ema20':None if pd.isna(r.ema20) else round(float(r.ema20),6),'ema50':None if pd.isna(r.ema50) else round(float(r.ema50),6),'bb_upper':None if pd.isna(r.bb_upper) else round(float(r.bb_upper),6),'bb_mid':None if pd.isna(r.bb_mid) else round(float(r.bb_mid),6),'bb_lower':None if pd.isna(r.bb_lower) else round(float(r.bb_lower),6),'buy':bool(r.buy),'sell':bool(r.sell)})
        return jsonify({'meta':series.get('meta',{}),'quote':quote,'strategy':strategy,'backtest':bt,'points':points})
    except Exception as e:return jsonify(error=str(e)),400

@app.get('/api/compare')
def compare():
    symbol=request.args.get('symbol','').strip().upper(); interval=request.args.get('interval','1day')
    if not symbol:return jsonify(error='Geen ticker opgegeven.'),400
    try:
        df=to_df(td_get('/time_series',{'symbol':symbol,'interval':interval,'outputsize':1000,'order':'ASC'}).get('values',[])); result=[]
        for s in STRATEGIES:
            r=backtest(df,s); result.append({'strategy':s,**{k:v for k,v in r.items() if k!='trade_list'},'score':strategy_rank(r)})
        result.sort(key=lambda x:x['score'],reverse=True); return jsonify(result)
    except Exception as e:return jsonify(error=str(e)),400

@app.get('/api/full-analysis')
def full_analysis():
    symbol=request.args.get('symbol','').strip().upper()
    if not symbol:return jsonify(error='Geen ticker opgegeven.'),400
    try:
        series=td_get('/time_series',{'symbol':symbol,'interval':'1day','outputsize':1200,'order':'ASC'}); quote=td_get('/quote',{'symbol':symbol}); df=to_df(series.get('values',[]))
        if len(df)<120:raise RuntimeError('Te weinig historische data voor een robuuste analyse.')
        cut=max(80,int(len(df)*0.70)); train=df.iloc[:cut].copy(); test=df.iloc[cut:].copy(); rows=[]
        for s in STRATEGIES:
            train_bt=backtest(train,s); test_bt=backtest(test,s); robust=round(0.35*strategy_rank(train_bt)+0.65*strategy_rank(test_bt),1)
            rows.append({'strategy':s,'score':robust,'train':train_bt,'test':test_bt})
        rows.sort(key=lambda x:x['score'],reverse=True); best=rows[0]
        tech,tech_reasons=technical_score(df); news=news_for(symbol); market_score=50
        try:
            spy=to_df(td_get('/time_series',{'symbol':'SPY','interval':'1day','outputsize':120,'order':'ASC'}).get('values',[])); market_score,_=technical_score(spy)
        except Exception: pass
        total=round(0.45*best['score']+0.30*tech+0.15*news['score']+0.10*market_score,1)
        verdict='POSITIEF' if total>=68 else ('VOORZICHTIG POSITIEF' if total>=58 else ('NEUTRAAL / WACHTEN' if total>=45 else 'NEGATIEF'))
        return jsonify({'symbol':symbol,'quote':quote,'best_strategy':best['strategy'],'strategy_score':best['score'],'technical_score':tech,'technical_reasons':tech_reasons,'news':news,'market_score':market_score,'total_score':total,'verdict':verdict,'strategies':rows,'method':'70% training / 30% out-of-sample; totaalscore combineert strategie, techniek, nieuws en markt.'})
    except Exception as e:return jsonify(error=str(e)),400

@app.get('/health')
def health():return jsonify(ok=True,api_key=bool(API_KEY),news_key=bool(NEWS_API_KEY))
if __name__=='__main__':app.run(host='0.0.0.0',port=int(os.getenv('PORT','80')))
