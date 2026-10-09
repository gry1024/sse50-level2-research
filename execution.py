"""Long-only buy execution timing: one pre-existing purchase mandate per stock/day."""
import json
from pathlib import Path
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from research import seconds,session,fee,boot
OUT=Path('results')
s=pl.read_parquet('sse50_20260923/snp_20260923.parquet');tt=seconds(s['Time'])
s=s.with_columns(pl.Series('time_seconds',tt)).join(pl.read_parquet(OUT/'features.parquet'),left_on=['Symbol','time_seconds'],right_on=['symbol','time_seconds'],how='inner').sort(['Symbol','Time'])
groups={int(k[0]):g for k,g in s.group_by('Symbol',maintain_order=True)}
from execution_core import depth_buy

def simulate(g,feature,direction,threshold,horizon,start,end):
 t=g['time_seconds'].to_numpy();a=g['AskPrice1'].to_numpy().astype(float);v=g['AskVolume1'].to_numpy();signal=g[feature].to_numpy()
 for i in np.flatnonzero((t>=start)&(t<end-horizon-6)&(direction*signal>=threshold)):
  j=np.searchsorted(t,t[i]+1);k=np.searchsorted(t,t[j]+horizon) if j<len(t) else len(t)
  if k>=len(t) or t[j]-t[i]>6 or t[k]-(t[j]+horizon)>6 or t[k]>end:continue
  q=int(min(10000/a[i],.01*v[i])/100)*100
  if q<100 or q>.1*v[j]:continue
  p0,ok0=depth_buy(g,j,q);p1,ok1=depth_buy(g,k,q)
  saving=q*(p0-p1)+fee(q*p0)-fee(q*p1)
  return dict(signal_time=float(t[i]),immediate_time=float(t[j]),delayed_time=float(t[k]),quantity=q,immediate_price=p0,delayed_price=p1,saving_cny=saving,saving_bps=saving/(q*p0)*1e4,notional=q*p0,exit_capacity=bool(ok0 and ok1),top1_shortfall=bool(q>.1*v[k]))
 return None
candidates=[]
for feature in ['obi1','obi5','trade_imbalance']:
 for direction in [-1,1]:
  for threshold in [.6,.8]:
   for horizon in [300,900,1800]:
    rs=[r for sym,g in groups.items() if sym%5!=0 for r in [simulate(g,feature,direction,threshold,horizon,34500,41400)] if r]
    score=sum(r['saving_cny'] for r in rs)/sum(r['notional'] for r in rs)*1e4 if rs else -1e9
    candidates.append(dict(feature=feature,direction=direction,threshold=threshold,horizon=horizon,trades=len(rs),failures=sum(not r['exit_capacity'] for r in rs),eligible=len(rs)>=10 and all(r['exit_capacity'] for r in rs),saving_bps=score))
candidates.sort(key=lambda r:(r['eligible'],r['saving_bps']),reverse=True);best=candidates[0];trades=[]
for split,start,end in [('morning',34500,41400),('afternoon',47100,53820)]:
 for sym,g in groups.items():
  r=simulate(g,**{k:best[k] for k in ['feature','direction','threshold','horizon']},start=start,end=end)
  if r:trades.append(dict(symbol=sym,split=split,heldout=sym%5==0,**r))
metrics={}
for name,rs in [('morning_train',[r for r in trades if r['split']=='morning' and not r['heldout']]),('afternoon_all',[r for r in trades if r['split']=='afternoon']),('afternoon_heldout',[r for r in trades if r['split']=='afternoon' and r['heldout']])]:
 total=sum(r['notional'] for r in rs)
 metrics[name]=dict(trades=len(rs),capacity_failures=sum(not r['exit_capacity'] for r in rs),saving_cny=sum(r['saving_cny'] for r in rs),saving_bps=sum(r['saving_cny'] for r in rs)/total*1e4 if total else None,win_rate=sum(r['saving_cny']>0 for r in rs)/len(rs) if rs else None,mean_trade_bps_ci95=boot([r['saving_bps'] for r in rs]))
summary=dict(selected=best,metrics=metrics,interpretation='Incremental purchase-cost saving versus immediate buy for an already intended identical quantity. Not investment return. Buy and hold, no intraday resale. Morning and afternoon are separate counterfactual experiments, not combined daily trading.',evaluation_status='Exploratory chronological split, not pristine out-of-sample: execution sizing and strategy family were revised after pilot afternoon inspection.')
(OUT/'execution_summary.json').write_text(json.dumps(summary,indent=2));pl.DataFrame(trades).write_csv(OUT/'execution_trades.csv');pl.DataFrame(candidates).write_csv(OUT/'execution_grid.csv')
rs=sorted([r for r in trades if r['split']=='afternoon'],key=lambda r:r['symbol']);fig,ax=plt.subplots(1,2,figsize=(12,4));ax[0].bar([str(r['symbol']) for r in rs],[r['saving_cny'] for r in rs],color=['#228866' if r['saving_cny']>0 else '#c34d50' for r in rs]);ax[0].tick_params(axis='x',rotation=90,labelsize=8);ax[0].set(title='Afternoon buy timing: savings vs immediate purchase',ylabel='CNY saved (same shares, one buy)');ax[1].bar(['Morning train','Afternoon all','Afternoon new stocks'],[metrics[k]['saving_bps'] or 0 for k in metrics]);ax[1].set(ylabel='Weighted savings (bps)',title='Exploratory chronology; not portfolio returns');fig.tight_layout();fig.savefig(OUT/'execution.png');plt.close(fig)
print(json.dumps(summary,indent=2))


