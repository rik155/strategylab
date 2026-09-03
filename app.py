import os
import math
import time
import json
import threading
from datetime import datetime, timezone
import requests
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from trading_bot import start_session, stop_session, session_status, broker

app = Flask(__name__, static_folder='.', static_url_path='')
API_KEY = os.getenv('TWELVE_DATA_API_KEY', '').strip()
NEWS_API_KEY = os.getenv('ALPHAVANTAGE_API_KEY', '').strip()
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '').strip()
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '').strip()
SCANNER_ENABLED = os.getenv('SCANNER_ENABLED', '1').strip().lower() not in ('0','false','no')
SCAN_INTERVAL_MINUTES = max(60, int(os.getenv('SCAN_INTERVAL_MINUTES', '60')))
SCAN_DELAY_SECONDS = max(8, int(os.getenv('SCAN_DELAY_SECONDS', '9')))
ALERT_MIN_SCORE = float(os.getenv('ALERT_MIN_SCORE', '70'))
BASE = 'https://api.twelvedata.com'
AV_BASE = 'https://www.alphavantage.co/query'
CONFIG_PATH = os.getenv('SCANNER_CONFIG', '/data/scanner_config.json')
STRATEGIES = ['ema','rsi','macd','bollinger','breakout']
DEFAULT_WATCHLIST = ['AAPL','MSFT','NVDA','AMD','AMZN','META','GOOGL','TSLA','PLTR','SOUN','ASML','AVGO']
PRESETS = {'tech':['AAPL','MSFT','NVDA','AMD','AVGO','QCOM','INTC','MU','ARM','AMZN','META','GOOGL','TSLA','PLTR','SOUN','CRWD','PANW','SNOW','NET','DDOG'],'us_large':['AAPL','MSFT','NVDA','AMZN','META','GOOGL','GOOG','BRK.B','LLY','AVGO','JPM','V','MA','WMT','XOM','COST','ORCL','NFLX','HD','PG','JNJ','ABBV','BAC','KO','CRM','CVX','MRK','AMD','PEP','TMO','MCD','ACN','CSCO','LIN','WFC','IBM','GE','CAT','DIS','UBER'],'growth':['NVDA','AMD','PLTR','SOUN','TSLA','CRWD','PANW','SNOW','NET','DDOG','SHOP','COIN','HOOD','RBLX','U','RKLB','IONQ','RIVN','SMCI','ARM'],'europe':['ASML','SAP','SIEGY','NVO','AZN','UL','SHEL','BP','GSK','DEO','ING','SAN','UBS','NVS','ERIC','NOK']}
scanner_state={'running':False,'last_scan':None,'last_error':None,'checked':0,'succeeded':0,'failed':0,'total':0,'matches':[],'errors':[],'next_scan':None,'current_symbol':None}
alert_history={};config_lock=threading.Lock()

def ensure_config_dir():
 d=os.path.dirname(CONFIG_PATH)
 if d:os.makedirs(d,exist_ok=True)
def load_config():
 with config_lock:
  ensure_config_dir()
  if os.path.exists(CONFIG_PATH):
   try:
    with open(CONFIG_PATH,'r',encoding='utf-8') as f:data=json.load(f)
    wl=[str(x).strip().upper() for x in data.get('watchlist',[]) if str(x).strip()]
    if wl:return {'watchlist':list(dict.fromkeys(wl))}
   except Exception:pass
  env=[x.strip().upper() for x in os.getenv('WATCHLIST','').split(',') if x.strip()];return {'watchlist':env or DEFAULT_WATCHLIST.copy()}
def save_config(cfg):
 with config_lock:
  ensure_config_dir();tmp=CONFIG_PATH+'.tmp'
  with open(tmp,'w',encoding='utf-8') as f:json.dump(cfg,f,indent=2)
  os.replace(tmp,CONFIG_PATH)
def watchlist():return load_config()['watchlist']
def is_rate_limit_message(data):
 if not isinstance(data,dict):return False
 text=' '.join(str(data.get(k,'')) for k in ('message','detail','Information','Note','status')).lower();return any(x in text for x in ('rate limit','api credits','credits','too many','limit reached','per minute'))
def td_get(path,params,retries=6):
 if not API_KEY:raise RuntimeError('TWELVE_DATA_API_KEY ontbreekt op de server.')
 params=dict(params);params['apikey']=API_KEY;last_error=None
 for attempt in range(retries):
  try:
   r=requests.get(BASE+path,params=params,timeout=25)
   if r.status_code==429:raise RuntimeError('Twelve Data rate limit bereikt')
   r.raise_for_status();data=r.json()
   if isinstance(data,dict) and data.get('status')=='error':raise RuntimeError(data.get('message','Twelve Data fout'))
   if is_rate_limit_message(data):raise RuntimeError('Twelve Data rate limit bereikt')
   return data
  except Exception as e:
   last_error=e;text=str(e).lower();limited=any(x in text for x in ('rate limit','credit','too many','429','per minute'))
   if not limited or attempt==retries-1:break
   wait=min(70,12*(attempt+1));scanner_state['last_error']=f'API-limiet bereikt; automatisch {wait}s wachten en daarna verder.';time.sleep(wait)
 raise RuntimeError(str(last_error or 'Twelve Data fout'))
def news_for(symbol):
 if not NEWS_API_KEY:return {'available':False,'score':50,'label':'Niet gekoppeld','items':[],'message':'Nieuws-API niet gekoppeld.'}
 try:
  r=requests.get(AV_BASE,params={'function':'NEWS_SENTIMENT','tickers':symbol,'limit':20,'sort':'LATEST','apikey':NEWS_API_KEY},timeout=20);r.raise_for_status();data=r.json()
  if data.get('Information') or data.get('Note'):return {'available':False,'score':50,'label':'Tijdelijk limiet','items':[],'message':data.get('Information') or data.get('Note')}
  feed=data.get('feed',[])[:12];items=[];scores=[]
  for item in feed:
   ts=next((z for z in item.get('ticker_sentiment',[]) if str(z.get('ticker','')).upper()==symbol.upper()),None)
   try:s=float((ts or {}).get('ticker_sentiment_score',item.get('overall_sentiment_score',0)))
   except Exception:s=0
   scores.append(s);items.append({'title':item.get('title',''),'source':item.get('source',''),'url':item.get('url',''),'time_published':item.get('time_published',''),'sentiment':round(s,3)})
  avg=float(np.mean(scores)) if scores else 0;score=max(0,min(100,50+avg*50));label='Positief' if score>=60 else ('Negatief' if score<=40 else 'Neutraal');return {'available':True,'score':round(score,1),'label':label,'items':items,'message':''}
 except Exception as e:return {'available':False,'score':50,'label':'Onbeschikbaar','items':[],'message':str(e)}
def to_df(values):
 df=pd.DataFrame(values)
 if df.empty:return df
 for c in ['open','high','low','close','volume']:
  if c in df.columns:df[c]=pd.to_numeric(df[c],errors='coerce')
 df['datetime']=pd.to_datetime(df['datetime']);return df.sort_values('datetime').reset_index(drop=True)
def ema(s,n):return s.ewm(span=n,adjust=False).mean()
def rsi(s,n=14):
 d=s.diff();up=d.clip(lower=0).rolling(n).mean();down=(-d.clip(upper=0)).rolling(n).mean();rs=up/down.replace(0,np.nan);return 100-(100/(1+rs))
def enrich(df):
 x=df.copy();x['ema20']=ema(x.close,20);x['ema50']=ema(x.close,50);x['ema12']=ema(x.close,12);x['ema26']=ema(x.close,26);x['macd']=x.ema12-x.ema26;x['macd_signal']=ema(x.macd,9);x['rsi']=rsi(x.close,14);mid=x.close.rolling(20).mean();sd=x.close.rolling(20).std();x['bb_mid']=mid;x['bb_upper']=mid+2*sd;x['bb_lower']=mid-2*sd;x['breakout_high']=x.high.shift(1).rolling(20).max();x['breakout_low']=x.low.shift(1).rolling(10).min();prev=x.close.shift(1);tr=pd.concat([(x.high-x.low).abs(),(x.high-prev).abs(),(x.low-prev).abs()],axis=1).max(axis=1);x['atr14']=tr.rolling(14).mean();return x
def signals(df,strategy):
 x=enrich(df);x['buy']=False;x['sell']=False
 if strategy=='ema':x['buy']=(x.ema20>x.ema50)&(x.ema20.shift(1)<=x.ema50.shift(1));x['sell']=(x.ema20<x.ema50)&(x.ema20.shift(1)>=x.ema50.shift(1))
 elif strategy=='rsi':x['buy']=(x.rsi>30)&(x.rsi.shift(1)<=30);x['sell']=(x.rsi<70)&(x.rsi.shift(1)>=70)
 elif strategy=='macd':x['buy']=(x.macd>x.macd_signal)&(x.macd.shift(1)<=x.macd_signal.shift(1));x['sell']=(x.macd<x.macd_signal)&(x.macd.shift(1)>=x.macd_signal.shift(1))
 elif strategy=='bollinger':x['buy']=(x.close>x.bb_lower)&(x.close.shift(1)<=x.bb_lower.shift(1));x['sell']=(x.close<x.bb_mid)&(x.close.shift(1)>=x.bb_mid.shift(1))
 elif strategy=='breakout':x['buy']=x.close>x.breakout_high;x['sell']=x.close<x.breakout_low
 return x
def backtest(df,strategy,fee_pct=.10):
 x=signals(df,strategy);in_pos=False;entry=0;entry_time=None;trades=[]
 for _,row in x.iterrows():
  if not in_pos and bool(row.buy) and pd.notna(row.close):in_pos=True;entry=float(row.close);entry_time=row.datetime
  elif in_pos and bool(row.sell) and pd.notna(row.close):
   exitp=float(row.close);trades.append({'entry_time':str(entry_time),'exit_time':str(row.datetime),'entry':entry,'exit':exitp,'return_pct':((exitp-entry)/entry*100)-(2*fee_pct)});in_pos=False
 if in_pos and len(x):
  exitp=float(x.iloc[-1].close);trades.append({'entry_time':str(entry_time),'exit_time':str(x.iloc[-1].datetime),'entry':entry,'exit':exitp,'return_pct':((exitp-entry)/entry*100)-(2*fee_pct)})
 returns=[t['return_pct'] for t in trades];wins=[r for r in returns if r>0];losses=[r for r in returns if r<=0];equity=peak=1;max_dd=0
 for rr in returns:equity*=1+rr/100;peak=max(peak,equity);max_dd=min(max_dd,(equity/peak-1)*100)
 gp=sum(wins);gl=abs(sum(losses));pf=gp/gl if gl else (999 if gp>0 else 0);return {'trades':len(trades),'winrate':round(len(wins)/len(trades)*100,1) if trades else 0,'total_return':round((equity-1)*100,2),'avg_trade':round(np.mean(returns),2) if returns else 0,'profit_factor':round(pf,2) if math.isfinite(pf) else 999,'max_drawdown':round(max_dd,2),'trade_list':trades[-20:]}
def strategy_rank(bt):
 trades=min(bt['trades'],30)/30;pf=min(bt['profit_factor'],3)/3;ret=max(-1,min(1,bt['total_return']/25));dd=max(0,1-min(abs(bt['max_drawdown']),25)/25);return round(100*(.35*((ret+1)/2)+.25*pf+.20*dd+.20*trades),1)
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
 x=enrich(df);r=x.iloc[-1];price=float(r.close);atr=float(r.atr14) if pd.notna(r.atr14) and r.atr14>0 else price*.025;look=x.tail(60);support=float(look.low.quantile(.15));resistance=float(look.high.quantile(.90));stop=max(support,price-1.6*atr);stop=min(stop,price*.985);risk=max(price-stop,price*.01);tp1=max(price+1.5*risk,resistance if resistance>price else price+1.5*risk);tp2=max(price+2.5*risk,tp1+.8*atr);latest=signals(df,best_strategy).iloc[-1];action='AFBOUWEN / VERKOOPSIGNAAL' if bool(latest.sell) or total_score<40 else ('WACHTEN / STRAKKE STOP' if total_score<55 else ('AANHOUDEN TOT DOELZONE' if total_score>=70 else 'AANHOUDEN / MONITOREN'));return {'current':round(price,4),'stop_loss':round(stop,4),'take_profit_1':round(tp1,4),'take_profit_2':round(tp2,4),'risk_pct':round((price-stop)/price*100,2),'reward1_pct':round((tp1-price)/price*100,2),'reward2_pct':round((tp2-price)/price*100,2),'rr1':round((tp1-price)/risk,2),'rr2':round((tp2-price)/risk,2),'action':action}
def analyze_df(symbol,df,include_news=True,market_score=50,quote=None):
 if len(df)<120:raise RuntimeError('Te weinig historische data voor een robuuste analyse.')
 cut=max(80,int(len(df)*.70));train=df.iloc[:cut].copy();test=df.iloc[cut:].copy();rows=[]
 for s in STRATEGIES:
  train_bt=backtest(train,s);test_bt=backtest(test,s);rows.append({'strategy':s,'score':round(.35*strategy_rank(train_bt)+.65*strategy_rank(test_bt),1),'train':train_bt,'test':test_bt})
 rows.sort(key=lambda x:x['score'],reverse=True);best=rows[0];tech,reasons=technical_score(df);news=news_for(symbol) if include_news else {'available':False,'score':50,'label':'Niet gescand','items':[],'message':''};total=round(.45*best['score']+.30*tech+.15*news['score']+.10*market_score,1);verdict='POSITIEF' if total>=68 else ('VOORZICHTIG POSITIEF' if total>=58 else ('NEUTRAAL / WACHTEN' if total>=45 else 'NEGATIEF'));return {'symbol':symbol,'quote':quote or {'symbol':symbol,'close':float(df.iloc[-1].close)},'best_strategy':best['strategy'],'strategy_score':best['score'],'technical_score':tech,'technical_reasons':reasons,'news':news,'market_score':market_score,'total_score':total,'verdict':verdict,'strategies':rows,'exit_plan':exit_plan(df,total,best['strategy'])}
def compute_full_analysis(symbol,include_news=True):
 series=td_get('/time_series',{'symbol':symbol,'interval':'1day','outputsize':1200,'order':'ASC'});quote=td_get('/quote',{'symbol':symbol});df=to_df(series.get('values',[]));market_score=50
 try:spy=to_df(td_get('/time_series',{'symbol':'SPY','interval':'1day','outputsize':120,'order':'ASC'}).get('values',[]));market_score,_=technical_score(spy)
 except Exception:pass
 return analyze_df(symbol,df,include_news,market_score,quote)
def compute_scanner_analysis(symbol,market_score):
 series=td_get('/time_series',{'symbol':symbol,'interval':'1day','outputsize':1200,'order':'ASC'});df=to_df(series.get('values',[]));return analyze_df(symbol,df,True,market_score,{'symbol':symbol,'close':float(df.iloc[-1].close) if len(df) else None})
def is_alert_candidate(a):
 best=a['strategies'][0]['test'];news_score=a['news']['score'] if a['news'].get('available') else 50;return a['total_score']>=ALERT_MIN_SCORE and a['technical_score']>=60 and a['market_score']>=50 and news_score>=45 and best['trades']>=6 and best['total_return']>0 and best['profit_factor']>=1.20 and a['exit_plan']['rr1']>=1.4
def telegram_send(text):
 if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:return False
 r=requests.post(f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage',json={'chat_id':TELEGRAM_CHAT_ID,'text':text,'disable_web_page_preview':True},timeout=20);r.raise_for_status();return True
def scan_once(send_alerts=True):
 wl=watchlist();scanner_state.update({'running':True,'last_error':None,'checked':0,'succeeded':0,'failed':0,'total':len(wl),'matches':[],'errors':[],'current_symbol':None});matches=[];market_score=50
 try:
  try:spy=to_df(td_get('/time_series',{'symbol':'SPY','interval':'1day','outputsize':120,'order':'ASC'}).get('values',[]));market_score,_=technical_score(spy)
  except Exception as e:scanner_state['last_error']=f'Marktfilter niet beschikbaar: {e}'
  for index,symbol in enumerate(wl,start=1):
   scanner_state['current_symbol']=symbol;scanner_state['checked']=index
   try:
    a=compute_scanner_analysis(symbol,market_score);scanner_state['succeeded']+=1;best=a['strategies'][0]['test'];p=a['exit_plan'];row={'symbol':symbol,'score':a['total_score'],'strategy':a['best_strategy'],'price':p['current'],'target1':p['take_profit_1'],'target2':p['take_profit_2'],'stop':p['stop_loss'],'oos_return':best['total_return'],'oos_trades':best['trades'],'news':a['news']['score'],'candidate':is_alert_candidate(a)};matches.append(row)
    if row['candidate']:
     last=alert_history.get(symbol,0);now=time.time()
     if send_alerts and now-last>12*3600:
      msg=f'🚨 StrategyLab kans: {symbol}\nScore: {a["total_score"]}/100 | Strategie: {a["best_strategy"].upper()}\nKoers: {p["current"]}\nDoel 1: {p["take_profit_1"]}\nDoel 2: {p["take_profit_2"]}\nStop: {p["stop_loss"]}\nNieuws: {a["news"]["label"]}';
      if telegram_send(msg):alert_history[symbol]=now
   except Exception as e:scanner_state['failed']+=1;err=f'{symbol}: {e}';scanner_state['last_error']=err;scanner_state['errors']=(scanner_state['errors']+[err])[-20:]
   if index<len(wl):time.sleep(SCAN_DELAY_SECONDS)
  matches.sort(key=lambda x:x['score'],reverse=True);scanner_state['matches']=matches;scanner_state['last_scan']=datetime.now(timezone.utc).isoformat()
 finally:scanner_state['running']=False;scanner_state['current_symbol']=None;scanner_state['next_scan']=datetime.fromtimestamp(time.time()+SCAN_INTERVAL_MINUTES*60,timezone.utc).isoformat()
 return matches
def scanner_loop():
 time.sleep(20)
 while True:
  if SCANNER_ENABLED:
   try:scan_once(True)
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
  series=td_get('/time_series',{'symbol':symbol,'interval':interval,'outputsize':outputsize,'order':'ASC'});quote=td_get('/quote',{'symbol':symbol});df=to_df(series.get('values',[]));x=signals(df,strategy);bt=backtest(df,strategy);points=[]
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
 try:return jsonify(compute_full_analysis(symbol,True))
 except Exception as e:return jsonify(error=str(e)),400
@app.get('/api/scanner/status')
def scanner_status():return jsonify({**scanner_state,'enabled':SCANNER_ENABLED,'interval_minutes':SCAN_INTERVAL_MINUTES,'scan_delay_seconds':SCAN_DELAY_SECONDS,'min_score':ALERT_MIN_SCORE,'watchlist':watchlist(),'telegram_ready':bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)})
@app.get('/api/scanner/config')
def scanner_config():return jsonify({'watchlist':watchlist(),'presets':PRESETS,'min_score':ALERT_MIN_SCORE,'interval_minutes':SCAN_INTERVAL_MINUTES})
@app.post('/api/scanner/watchlist')
def scanner_watchlist_update():
 d=request.get_json(silent=True) or {};action=d.get('action');symbol=str(d.get('symbol','')).strip().upper();cfg=load_config();wl=cfg['watchlist']
 if action=='add' and symbol and symbol not in wl:wl.append(symbol)
 elif action=='remove' and symbol in wl:wl.remove(symbol)
 elif action=='clear':wl=[]
 elif action=='preset':wl=list(dict.fromkeys(wl+PRESETS.get(str(d.get('preset','')),[])))
 elif action=='replace':wl=[str(x).strip().upper() for x in d.get('watchlist',[]) if str(x).strip()]
 else:return jsonify(ok=False,message='Ongeldige actie.'),400
 save_config({'watchlist':wl});return jsonify(ok=True,watchlist=wl)
@app.post('/api/scanner/run')
def scanner_run():
 if scanner_state['running']:return jsonify(ok=False,message='Scanner draait al.'),409
 threading.Thread(target=scan_once,kwargs={'send_alerts':True},daemon=True).start();return jsonify(ok=True,message=f'Scanner gestart voor alle {len(watchlist())} aandelen in de watchlist.')
@app.post('/api/scanner/test-alert')
def scanner_test_alert():
 try:telegram_send('✅ StrategyLab testmelding werkt.');return jsonify(ok=True)
 except Exception as e:return jsonify(ok=False,message=str(e)),400
@app.get('/api/trading/status')
def trading_status():return jsonify({'session':session_status(),'broker':broker.status()})
@app.post('/api/trading/start')
def trading_start():
 d=request.get_json(silent=True) or {}
 try:
  s=start_session(d.get('budget'),d.get('hours'),d.get('max_loss'),d.get('risk_per_trade_pct',1));telegram_send(f'🤖 StrategyLab sessie gestart\nBudget: €{s["budget"]}\nLooptijd tot: {s["ends_at"]}\nMax verlies: €{s["max_loss"]}\nModus: SIMULATIE — IBKR nog niet gekoppeld');return jsonify(ok=True,session=s,broker=broker.status())
 except Exception as e:return jsonify(ok=False,message=str(e)),400
@app.post('/api/trading/stop')
def trading_stop():
 d=request.get_json(silent=True) or {};s=stop_session(bool(d.get('emergency',False)));telegram_send('🛑 StrategyLab trading sessie gestopt.');return jsonify(ok=True,session=s)
@app.get('/health')
def health():return jsonify(ok=True,api_key=bool(API_KEY),news_key=bool(NEWS_API_KEY),telegram=bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),scanner=SCANNER_ENABLED,broker=broker.status())
if __name__=='__main__':
 threading.Thread(target=scanner_loop,daemon=True).start();app.run(host='0.0.0.0',port=int(os.getenv('PORT','80')),threaded=True)
