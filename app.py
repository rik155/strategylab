import os
import math
import time
import threading
from datetime import datetime, timezone
import requests
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder='.', static_url_path='')
API_KEY = os.getenv('TWELVE_DATA_API_KEY', '').strip()
NEWS_API_KEY = os.getenv('ALPHAVANTAGE_API_KEY', '').strip()
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '').strip()
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '').strip()
SCANNER_ENABLED = os.getenv('SCANNER_ENABLED', '1').strip().lower() not in ('0','false','no')
SCAN_INTERVAL_MINUTES = max(60, int(os.getenv('SCAN_INTERVAL_MINUTES', '60')))
ALERT_MIN_SCORE = float(os.getenv('ALERT_MIN_SCORE', '70'))
WATCHLIST = [x.strip().upper() for x in os.getenv('WATCHLIST', 'AAPL,MSFT,NVDA,AMD,AMZN,META,GOOGL,TSLA,PLTR,SOUN,ASML,AVGO').split(',') if x.strip()]
BASE = 'https://api.twelvedata.com'
AV_BASE = 'https://www.alphavantage.co/query'
STRATEGIES = ['ema','rsi','macd','bollinger','breakout']
scanner_state = {'running': False, 'last_scan': None, 'last_error': None, 'checked': 0, 'matches': [], 'next_scan': None}
alert_history = {}


def td_get(path, params):
    if not API_KEY:
        raise RuntimeError('TWELVE_DATA_API_KEY ontbreekt op de server.')
    params = dict(params); params['apikey'] = API_KEY
    r = requests.get(BASE + path, params=params, timeout=20); r.raise_for_status(); data = r.json()
    if isinstance(data, dict) and data.get('status') == 'error': raise RuntimeError(data.get('message', 'Twelve Data fout'))
    return data


def news_for(symbol):
    if not NEWS_API_KEY:
        return {'available': False, 'score': 50, 'label': 'Niet gekoppeld', 'items': [], 'message': 'Voeg ALPHAVANTAGE_API_KEY toe voor nieuws en sentiment.'}
    try:
        r=requests.get(AV_BASE,params={'function':'NEWS_SENTIMENT','tickers':symbol,'limit':20,'sort':'LATEST','apikey':NEWS_API_KEY},timeout=20); r.raise_for_status(); data=r.json(); feed=data.get('feed',[])[:12]; items=[]; scores=[]
        for item in feed:
            ticker_score=None
            for ts in item.get('ticker_sentiment',[]):
                if str(ts.get('ticker','')).upper()==symbol.upper():
                    try:ticker_score=float(ts.get('ticker_sentiment_score'))
                    except Exception:ticker_score=None
                    break
            if ticker_score is None:
                try:ticker_score=float(item.get('overall_sentiment_score',0))
                except Exception:ticker_score=0
            scores.append(ticker_score); items.append({'title':item.get('title',''),'source':item.get('source',''),'url':item.get('url',''),'time_published':item.get('time_published',''),'sentiment':round(ticker_score,3)})
        avg=float(np.mean(scores)) if scores else 0; score=max(0,min(100,50+avg*50)); label='Positief' if score>=60 else ('Negatief' if score<=40 else 'Neutraal')
        return {'available':True,'score':round(score,1),'label':label,'items':items,'message':''}
    except Exception as e:return {'available':False,'score':50,'label':'Onbeschikbaar','items':[],'message':str(e)}


def to_df(values):
    df=pd.DataFrame(values)
    if df.empty:return df
    for c in ['open','high','low','close','volume']:
        if c in df.columns:df[c]=pd.to_numeric(df[c],errors='coerce')
    df['datetime']=pd.to_datetime(df['datetime']); return df.sort_values('datetime').reset_index(drop=True)


def ema(s,n):return s.ewm(span=n,adjust=False).mean()
def rsi(s,n=14):
    d=s.diff(); up=d.clip(lower=0).rolling(n).mean(); down=(-d.clip(upper=0)).rolling(n).mean(); rs=up/down.replace(0,np.nan); return 100-(100/(1+rs))


def enrich(df):
    x=df.copy(); x['ema20']=ema(x.close,20); x['ema50']=ema(x.close,50); x['ema12']=ema(x.close,12); x['ema26']=ema(x.close,26); x['macd']=x.ema12-x.ema26; x['macd_signal']=ema(x.macd,9); x['rsi']=rsi(x.close,14)
    mid=x.close.rolling(20).mean(); sd=x.close.rolling(20).std(); x['bb_mid']=mid; x['bb_upper']=mid+2*sd; x['bb_lower']=mid-2*sd; x['breakout_high']=x.high.shift(1).rolling(20).max(); x['breakout_low']=x.low.shift(1).rolling(10).min()
    prev=x.close.shift(1); tr=pd.concat([(x.high-x.low).abs(),(x.high-prev).abs(),(x.low-prev).abs()],axis=1).max(axis=1); x['atr14']=tr.rolling(14).mean(); return x


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
        if not in_pos and bool(row.buy) and pd.notna(row.close):in_pos=True;entry=float(row.close);entry_time=row.datetime
        elif in_pos and bool(row.sell) and pd.notna(row.close):
            exitp=float(row.close);net=((exitp-entry)/entry*100)-(2*fee_pct);trades.append({'entry_time':str(entry_time),'exit_time':str(row.datetime),'entry':entry,'exit':exitp,'return_pct':net});in_pos=False
    if in_pos and len(x):
        exitp=float(x.iloc[-1].close);net=((exitp-entry)/entry*100)-(2*fee_pct);trades.append({'entry_time':str(entry_time),'exit_time':str(x.iloc[-1].datetime),'entry':entry,'exit':exitp,'return_pct':net})
    returns=[t['return_pct'] for t in trades];wins=[r for r in returns if r>0];losses=[r for r in returns if r<=0];equity=1.0;peak=1.0;max_dd=0.0
    for r in returns:equity*=1+r/100;peak=max(peak,equity);max_dd=min(max_dd,(equity/peak-1)*100)
    gp=sum(wins);gl=abs(sum(losses));pf=gp/gl if gl>0 else (999 if gp>0 else 0)
    return {'trades':len(trades),'winrate':round(len(wins)/len(trades)*100,1) if trades else 0,'total_return':round((equity-1)*100,2),'avg_trade':round(np.mean(returns),2) if returns else 0,'profit_factor':round(pf,2) if math.isfinite(pf) else 999,'max_drawdown':round(max_dd,2),'trade_list':trades[-20:]}


def strategy_rank(bt):
    trades=min(bt['trades'],30)/30;pf=min(bt['profit_factor'],3)/3;ret=max(-1,min(1,bt['total_return']/25));dd=max(0,1-min(abs(bt['max_drawdown']),25)/25)
    return round(100*(0.35*((ret+1)/2)+0.25*pf+0.20*dd+0.20*trades),1)


def technical_score(df):
    x=enrich(df);r=x.iloc[-1];score=50;reasons=[]
    if pd.notna(r.ema20) and pd.notna(r.ema50):
        if r.ema20>r.ema50:score+=15;reasons.append('Korte trend ligt boven lange trend')
        else:score-=15;reasons.append('Korte trend ligt onder lange trend')
    if pd.notna(r.rsi):
        if 45<=r.rsi<=65:score+=10;reasons.append('RSI ligt in een gezonde positieve zone')
        elif r.rsi>75:score-=10;reasons.append('RSI is sterk overbought')
        elif r.rsi<30:score-=5;reasons.append('RSI is oversold')
    if pd.notna(r.macd) and pd.notna(r.macd_signal):
        if r.macd>r.macd_signal:score+=10;reasons.append('MACD is positief')
        else:score-=10;reasons.append('MACD is negatief')
    if pd.notna(r.breakout_high) and r.close>r.breakout_high:score+=10;reasons.append('Koers breekt boven recente weerstand')
    return max(0,min(100,score)),reasons


def exit_plan(df,total_score,best_strategy):
    x=enrich(df);r=x.iloc[-1];price=float(r.close);atr=float(r.atr14) if pd.notna(r.atr14) and r.atr14>0 else price*0.025
    look=x.tail(60);support=float(look.low.quantile(0.15));resistance=float(look.high.quantile(0.90))
    stop=max(support,price-1.6*atr); stop=min(stop,price*0.985); risk=max(price-stop,price*0.01)
    tp1=max(price+1.5*risk,resistance if resistance>price else price+1.5*risk);tp2=max(price+2.5*risk,tp1+0.8*atr)
    latest=signals(df,best_strategy).iloc[-1]
    if bool(latest.sell) or total_score<40:action='AFBOUWEN / VERKOOPSIGNAAL'
    elif total_score<55:action='WACHTEN / STRAKKE STOP'
    elif total_score>=70:action='AANHOUDEN TOT DOELZONE'
    else:action='AANHOUDEN / MONITOREN'
    return {'current':round(price,4),'stop_loss':round(stop,4),'take_profit_1':round(tp1,4),'take_profit_2':round(tp2,4),'resistance':round(resistance,4),'atr14':round(atr,4),'risk_pct':round((price-stop)/price*100,2),'reward1_pct':round((tp1-price)/price*100,2),'reward2_pct':round((tp2-price)/price*100,2),'rr1':round((tp1-price)/risk,2),'rr2':round((tp2-price)/risk,2),'action':action}


def compute_full_analysis(symbol, include_news=True):
    series=td_get('/time_series',{'symbol':symbol,'interval':'1day','outputsize':1200,'order':'ASC'});quote=td_get('/quote',{'symbol':symbol});df=to_df(series.get('values',[]))
    if len(df)<120:raise RuntimeError('Te weinig historische data voor een robuuste analyse.')
    cut=max(80,int(len(df)*0.70));train=df.iloc[:cut].copy();test=df.iloc[cut:].copy();rows=[]
    for s in STRATEGIES:
        train_bt=backtest(train,s);test_bt=backtest(test,s);robust=round(0.35*strategy_rank(train_bt)+0.65*strategy_rank(test_bt),1);rows.append({'strategy':s,'score':robust,'train':train_bt,'test':test_bt})
    rows.sort(key=lambda x:x['score'],reverse=True);best=rows[0];tech,tech_reasons=technical_score(df);news=news_for(symbol) if include_news else {'available':False,'score':50,'label':'Niet gescand','items':[],'message':''};market_score=50
    try:spy=to_df(td_get('/time_series',{'symbol':'SPY','interval':'1day','outputsize':120,'order':'ASC'}).get('values',[]));market_score,_=technical_score(spy)
    except Exception:pass
    total=round(0.45*best['score']+0.30*tech+0.15*news['score']+0.10*market_score,1);verdict='POSITIEF' if total>=68 else ('VOORZICHTIG POSITIEF' if total>=58 else ('NEUTRAAL / WACHTEN' if total>=45 else 'NEGATIEF'));plan=exit_plan(df,total,best['strategy'])
    return {'symbol':symbol,'quote':quote,'best_strategy':best['strategy'],'strategy_score':best['score'],'technical_score':tech,'technical_reasons':tech_reasons,'news':news,'market_score':market_score,'total_score':total,'verdict':verdict,'strategies':rows,'exit_plan':plan,'method':'70% training / 30% out-of-sample; totaalscore combineert strategie, techniek, nieuws en markt.'}


def is_alert_candidate(a):
    best=a['strategies'][0]['test']; news_score=a['news']['score'] if a['news'].get('available') else 50
    return (a['total_score']>=ALERT_MIN_SCORE and a['technical_score']>=60 and a['market_score']>=50 and news_score>=45 and best['trades']>=6 and best['total_return']>0 and best['profit_factor']>=1.20 and a['exit_plan']['rr1']>=1.4)


def telegram_send(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:return False
    r=requests.post(f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage',json={'chat_id':TELEGRAM_CHAT_ID,'text':text,'disable_web_page_preview':True},timeout=20)
    r.raise_for_status();return True


def scan_once(send_alerts=True):
    scanner_state['running']=True;scanner_state['last_error']=None;scanner_state['checked']=0;matches=[]
    try:
        for symbol in WATCHLIST:
            try:
                a=compute_full_analysis(symbol,include_news=True);scanner_state['checked']+=1
                if is_alert_candidate(a):
                    p=a['exit_plan'];best=a['strategies'][0]['test'];matches.append({'symbol':symbol,'score':a['total_score'],'strategy':a['best_strategy'],'price':p['current'],'target1':p['take_profit_1'],'target2':p['take_profit_2'],'stop':p['stop_loss'],'oos_return':best['total_return'],'oos_trades':best['trades'],'news':a['news']['score']})
                    last=alert_history.get(symbol,0);now=time.time()
                    if send_alerts and now-last>12*3600:
                        msg=(f'🚨 StrategyLab kans: {symbol}\n'
                             f'Score: {a["total_score"]}/100 | Strategie: {a["best_strategy"].upper()}\n'
                             f'Koers: {p["current"]}\n'
                             f'Doel 1: {p["take_profit_1"]} (+{p["reward1_pct"]}%)\n'
                             f'Doel 2: {p["take_profit_2"]} (+{p["reward2_pct"]}%)\n'
                             f'Stop: {p["stop_loss"]} (-{p["risk_pct"]}%)\n'
                             f'OOS: {best["total_return"]}% | {best["trades"]} trades | PF {best["profit_factor"]}\n'
                             f'Nieuws: {a["news"]["label"]} ({a["news"]["score"]}/100)\n'
                             f'Geen financieel advies; controleer zelf voor je handelt.')
                        if telegram_send(msg):alert_history[symbol]=now
                time.sleep(2)
            except Exception as e:
                scanner_state['last_error']=f'{symbol}: {e}'
        matches.sort(key=lambda x:x['score'],reverse=True);scanner_state['matches']=matches[:20];scanner_state['last_scan']=datetime.now(timezone.utc).isoformat()
    finally:
        scanner_state['running']=False;scanner_state['next_scan']=datetime.fromtimestamp(time.time()+SCAN_INTERVAL_MINUTES*60,timezone.utc).isoformat()
    return matches


def scanner_loop():
    time.sleep(20)
    while True:
        if SCANNER_ENABLED:
            try:scan_once(send_alerts=True)
            except Exception as e:scanner_state['last_error']=str(e);scanner_state['running']=False
        time.sleep(SCAN_INTERVAL_MINUTES*60)


@app.get('/')
def home():return send_from_directory('.','index.html')
@app.get('/api/search')
def search():
    q=request.args.get('q','').strip()
    if not q:return jsonify([])
    try:return jsonify(td_get('/symbol_search',{'symbol':q,'outputsize':8}).get('data',[]))
    except Exception as e:return jsonify(error=str(e)),400
@app.get('/api/analyze')
def analyze():
    symbol=request.args.get('symbol','').strip().upper();interval=request.args.get('interval','1day');strategy=request.args.get('strategy','ema');outputsize=min(int(request.args.get('outputsize','500')),5000)
    if not symbol:return jsonify(error='Geen ticker opgegeven.'),400
    try:
        series=td_get('/time_series',{'symbol':symbol,'interval':interval,'outputsize':outputsize,'order':'ASC'});quote=td_get('/quote',{'symbol':symbol});df=to_df(series.get('values',[]))
        if df.empty:raise RuntimeError('Geen koersdata gevonden.')
        x=signals(df,strategy);bt=backtest(df,strategy);points=[]
        for _,r in x.iterrows():points.append({'time':r.datetime.isoformat(),'open':None if pd.isna(r.open) else round(float(r.open),6),'high':None if pd.isna(r.high) else round(float(r.high),6),'low':None if pd.isna(r.low) else round(float(r.low),6),'close':None if pd.isna(r.close) else round(float(r.close),6),'volume':None if 'volume' not in x.columns or pd.isna(r.volume) else float(r.volume),'ema20':None if pd.isna(r.ema20) else round(float(r.ema20),6),'ema50':None if pd.isna(r.ema50) else round(float(r.ema50),6),'bb_upper':None if pd.isna(r.bb_upper) else round(float(r.bb_upper),6),'bb_mid':None if pd.isna(r.bb_mid) else round(float(r.bb_mid),6),'bb_lower':None if pd.isna(r.bb_lower) else round(float(r.bb_lower),6),'buy':bool(r.buy),'sell':bool(r.sell)})
        return jsonify({'meta':series.get('meta',{}),'quote':quote,'strategy':strategy,'backtest':bt,'points':points})
    except Exception as e:return jsonify(error=str(e)),400
@app.get('/api/compare')
def compare():
    symbol=request.args.get('symbol','').strip().upper();interval=request.args.get('interval','1day')
    if not symbol:return jsonify(error='Geen ticker opgegeven.'),400
    try:
        df=to_df(td_get('/time_series',{'symbol':symbol,'interval':interval,'outputsize':1000,'order':'ASC'}).get('values',[]));result=[]
        for s in STRATEGIES:
            r=backtest(df,s);result.append({'strategy':s,**{k:v for k,v in r.items() if k!='trade_list'},'score':strategy_rank(r)})
        result.sort(key=lambda x:x['score'],reverse=True);return jsonify(result)
    except Exception as e:return jsonify(error=str(e)),400
@app.get('/api/full-analysis')
def full_analysis():
    symbol=request.args.get('symbol','').strip().upper()
    if not symbol:return jsonify(error='Geen ticker opgegeven.'),400
    try:return jsonify(compute_full_analysis(symbol,include_news=True))
    except Exception as e:return jsonify(error=str(e)),400
@app.get('/api/scanner/status')
def scanner_status():
    return jsonify({**scanner_state,'enabled':SCANNER_ENABLED,'interval_minutes':SCAN_INTERVAL_MINUTES,'min_score':ALERT_MIN_SCORE,'watchlist':WATCHLIST,'telegram_ready':bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)})
@app.post('/api/scanner/run')
def scanner_run():
    if scanner_state['running']:return jsonify(ok=False,message='Scanner draait al.'),409
    threading.Thread(target=scan_once,kwargs={'send_alerts':True},daemon=True).start();return jsonify(ok=True,message='Scanner gestart.')
@app.post('/api/scanner/test-alert')
def scanner_test_alert():
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:return jsonify(ok=False,message='Telegram is nog niet ingesteld.'),400
    try:telegram_send('✅ StrategyLab testmelding werkt. Je ontvangt vanaf nu alleen alerts die aan de ingestelde kwaliteitsfilters voldoen.');return jsonify(ok=True)
    except Exception as e:return jsonify(ok=False,message=str(e)),400
@app.get('/health')
def health():return jsonify(ok=True,api_key=bool(API_KEY),news_key=bool(NEWS_API_KEY),telegram=bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),scanner=SCANNER_ENABLED)

if __name__=='__main__':
    if os.getenv('WERKZEUG_RUN_MAIN') != 'true':threading.Thread(target=scanner_loop,daemon=True).start()
    app.run(host='0.0.0.0',port=int(os.getenv('PORT','80')),threaded=True)
