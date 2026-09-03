import json
import os
import threading
import time
from datetime import datetime, timezone

STATE_PATH = os.getenv('TRADING_STATE', '/data/trading_state.json')
_lock = threading.Lock()


def _now(): return datetime.now(timezone.utc).isoformat()

def _blank():
    return {'active':False,'mode':'simulation','broker':'IBKR_PENDING','started_at':None,'ends_at':None,'budget':0.0,'max_loss':0.0,'risk_per_trade_pct':1.0,'cash':0.0,'realized_pnl':0.0,'unrealized_pnl':0.0,'positions':[],'trades':[],'last_action':'Nog geen sessie gestart','emergency_stop':False}

def load_state():
    with _lock:
        try:
            with open(STATE_PATH,'r',encoding='utf-8') as f:return {**_blank(),**json.load(f)}
        except Exception:return _blank()

def save_state(s):
    with _lock:
        os.makedirs(os.path.dirname(STATE_PATH),exist_ok=True)
        tmp=STATE_PATH+'.tmp'
        with open(tmp,'w',encoding='utf-8') as f:json.dump(s,f,indent=2)
        os.replace(tmp,STATE_PATH)

def start_session(budget,hours,max_loss,risk_per_trade_pct=1.0):
    budget=float(budget);hours=float(hours);max_loss=float(max_loss);risk=float(risk_per_trade_pct)
    if budget<=0 or hours<=0 or max_loss<=0:raise ValueError('Bedrag, looptijd en maximaal verlies moeten groter dan 0 zijn.')
    if max_loss>=budget:raise ValueError('Maximaal verlies moet lager zijn dan het sessiebudget.')
    if not 0.1<=risk<=5:raise ValueError('Risico per trade moet tussen 0,1% en 5% liggen.')
    now=time.time();s=_blank();s.update({'active':True,'started_at':_now(),'ends_at':datetime.fromtimestamp(now+hours*3600,timezone.utc).isoformat(),'budget':round(budget,2),'max_loss':round(max_loss,2),'risk_per_trade_pct':risk,'cash':round(budget,2),'last_action':'Trading sessie gestart in simulatiemodus'})
    save_state(s);return s

def stop_session(emergency=False):
    s=load_state();s['active']=False;s['emergency_stop']=bool(emergency);s['last_action']='NOODSTOP geactiveerd' if emergency else 'Trading sessie gestopt';save_state(s);return s

def session_status():
    s=load_state()
    if s['active'] and s.get('ends_at'):
        try:
            if datetime.now(timezone.utc)>=datetime.fromisoformat(s['ends_at']):
                s['active']=False;s['last_action']='Sessietijd verstreken';save_state(s)
        except Exception:pass
    if s['active'] and s['realized_pnl']+s['unrealized_pnl']<=-abs(s['max_loss']):
        s['active']=False;s['emergency_stop']=True;s['last_action']='Automatisch gestopt: maximale verliesgrens bereikt';save_state(s)
    return s

class BrokerAdapter:
    """Interface for IBKR. Real order placement stays disabled until credentials/connectivity are configured."""
    def status(self):return {'connected':False,'name':'IBKR','live_orders_enabled':False,'message':'IBKR nog niet gekoppeld'}
    def place_order(self,*args,**kwargs):raise RuntimeError('Live orders zijn nog niet ingeschakeld.')
    def close_position(self,*args,**kwargs):raise RuntimeError('Live orders zijn nog niet ingeschakeld.')

broker=BrokerAdapter()
