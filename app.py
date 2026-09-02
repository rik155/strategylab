import os
import math
import requests
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder='.', static_url_path='')
API_KEY = os.getenv('TWELVE_DATA_API_KEY', '').strip()
BASE = 'https://api.twelvedata.com'


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


def to_df(values):
    df = pd.DataFrame(values)
    if df.empty:
        return df
    for c in ['open','high','low','close','volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df.sort_values('datetime').reset_index(drop=True)


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    down = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def enrich(df):
    x = df.copy()
    x['ema20'] = ema(x.close, 20)
    x['ema50'] = ema(x.close, 50)
    x['ema12'] = ema(x.close, 12)
    x['ema26'] = ema(x.close, 26)
    x['macd'] = x.ema12 - x.ema26
    x['macd_signal'] = ema(x.macd, 9)
    x['rsi'] = rsi(x.close, 14)
    mid = x.close.rolling(20).mean()
    sd = x.close.rolling(20).std()
    x['bb_mid'] = mid
    x['bb_upper'] = mid + 2*sd
    x['bb_lower'] = mid - 2*sd
    x['breakout_high'] = x.high.shift(1).rolling(20).max()
    x['breakout_low'] = x.low.shift(1).rolling(10).min()
    return x


def signals(df, strategy):
    x = enrich(df)
    x['buy'] = False
    x['sell'] = False
    if strategy == 'ema':
        x['buy'] = (x.ema20 > x.ema50) & (x.ema20.shift(1) <= x.ema50.shift(1))
        x['sell'] = (x.ema20 < x.ema50) & (x.ema20.shift(1) >= x.ema50.shift(1))
    elif strategy == 'rsi':
        x['buy'] = (x.rsi > 30) & (x.rsi.shift(1) <= 30)
        x['sell'] = (x.rsi < 70) & (x.rsi.shift(1) >= 70)
    elif strategy == 'macd':
        x['buy'] = (x.macd > x.macd_signal) & (x.macd.shift(1) <= x.macd_signal.shift(1))
        x['sell'] = (x.macd < x.macd_signal) & (x.macd.shift(1) >= x.macd_signal.shift(1))
    elif strategy == 'bollinger':
        x['buy'] = (x.close > x.bb_lower) & (x.close.shift(1) <= x.bb_lower.shift(1))
        x['sell'] = (x.close < x.bb_mid) & (x.close.shift(1) >= x.bb_mid.shift(1))
    elif strategy == 'breakout':
        x['buy'] = x.close > x.breakout_high
        x['sell'] = x.close < x.breakout_low
    return x


def backtest(df, strategy, fee_pct=0.10):
    x = signals(df, strategy)
    in_pos = False
    entry = 0.0
    entry_time = None
    trades = []
    for _, row in x.iterrows():
        if not in_pos and bool(row.buy) and pd.notna(row.close):
            in_pos = True
            entry = float(row.close)
            entry_time = row.datetime
        elif in_pos and bool(row.sell) and pd.notna(row.close):
            exitp = float(row.close)
            gross = (exitp-entry)/entry*100
            net = gross - (2*fee_pct)
            trades.append({'entry_time': str(entry_time), 'exit_time': str(row.datetime), 'entry': entry, 'exit': exitp, 'return_pct': net})
            in_pos = False
    if in_pos and len(x):
        exitp = float(x.iloc[-1].close)
        gross = (exitp-entry)/entry*100
        net = gross - (2*fee_pct)
        trades.append({'entry_time': str(entry_time), 'exit_time': str(x.iloc[-1].datetime), 'entry': entry, 'exit': exitp, 'return_pct': net})

    returns = [t['return_pct'] for t in trades]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in returns:
        equity *= (1 + r/100)
        peak = max(peak, equity)
        dd = (equity/peak - 1)*100
        max_dd = min(max_dd, dd)
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit/gross_loss if gross_loss > 0 else (999 if gross_profit > 0 else 0)
    return {
        'trades': len(trades),
        'winrate': round((len(wins)/len(trades)*100),1) if trades else 0,
        'total_return': round((equity-1)*100,2),
        'avg_trade': round(np.mean(returns),2) if returns else 0,
        'profit_factor': round(profit_factor,2) if math.isfinite(profit_factor) else 999,
        'max_drawdown': round(max_dd,2),
        'trade_list': trades[-20:]
    }


@app.get('/')
def home():
    return send_from_directory('.', 'index.html')

@app.get('/api/search')
def search():
    q = request.args.get('q','').strip()
    if not q:
        return jsonify([])
    try:
        d = td_get('/symbol_search', {'symbol': q, 'outputsize': 8})
        return jsonify(d.get('data', []))
    except Exception as e:
        return jsonify(error=str(e)), 400

@app.get('/api/analyze')
def analyze():
    symbol = request.args.get('symbol','').strip().upper()
    interval = request.args.get('interval','1day')
    strategy = request.args.get('strategy','ema')
    outputsize = min(int(request.args.get('outputsize','500')), 5000)
    if not symbol:
        return jsonify(error='Geen ticker opgegeven.'), 400
    try:
        series = td_get('/time_series', {'symbol': symbol, 'interval': interval, 'outputsize': outputsize, 'order': 'ASC'})
        quote = td_get('/quote', {'symbol': symbol})
        df = to_df(series.get('values', []))
        if df.empty:
            raise RuntimeError('Geen koersdata gevonden.')
        x = signals(df, strategy)
        bt = backtest(df, strategy)
        points = []
        for _, r in x.iterrows():
            points.append({
                'time': r.datetime.isoformat(),
                'open': None if pd.isna(r.open) else round(float(r.open),6),
                'high': None if pd.isna(r.high) else round(float(r.high),6),
                'low': None if pd.isna(r.low) else round(float(r.low),6),
                'close': None if pd.isna(r.close) else round(float(r.close),6),
                'volume': None if 'volume' not in x.columns or pd.isna(r.volume) else float(r.volume),
                'ema20': None if pd.isna(r.ema20) else round(float(r.ema20),6),
                'ema50': None if pd.isna(r.ema50) else round(float(r.ema50),6),
                'bb_upper': None if pd.isna(r.bb_upper) else round(float(r.bb_upper),6),
                'bb_mid': None if pd.isna(r.bb_mid) else round(float(r.bb_mid),6),
                'bb_lower': None if pd.isna(r.bb_lower) else round(float(r.bb_lower),6),
                'buy': bool(r.buy),
                'sell': bool(r.sell)
            })
        return jsonify({
            'meta': series.get('meta', {}),
            'quote': quote,
            'strategy': strategy,
            'backtest': bt,
            'points': points
        })
    except Exception as e:
        return jsonify(error=str(e)), 400

@app.get('/api/compare')
def compare():
    symbol = request.args.get('symbol','').strip().upper()
    interval = request.args.get('interval','1day')
    if not symbol:
        return jsonify(error='Geen ticker opgegeven.'), 400
    try:
        series = td_get('/time_series', {'symbol': symbol, 'interval': interval, 'outputsize': 1000, 'order': 'ASC'})
        df = to_df(series.get('values', []))
        result=[]
        for s in ['ema','rsi','macd','bollinger','breakout']:
            r = backtest(df, s)
            result.append({'strategy':s, **{k:v for k,v in r.items() if k!='trade_list'}})
        result.sort(key=lambda x: x['total_return'], reverse=True)
        return jsonify(result)
    except Exception as e:
        return jsonify(error=str(e)), 400

@app.get('/health')
def health():
    return jsonify(ok=True, api_key=bool(API_KEY))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT','80')))
