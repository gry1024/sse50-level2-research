"""Reproducible single-day Level-2 research; prices in CNY, time in seconds."""
from pathlib import Path
import argparse, json, hashlib, math
import numpy as np
import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('results'); OUT.mkdir(exist_ok=True)
plt.rcParams.update({'figure.dpi':140,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2})

def seconds(a):
 a=np.asarray(a,dtype=np.int64)
 return a//10000000*3600+(a//100000%100)*60+(a//1000%100)+(a%1000)/1000

def session(t):
 return ((t>=34200)&(t<41400))|((t>=46800)&(t<53820))

def fee(notional,sell=False):
 return max(5.,notional*.0002)+notional*.00001+(notional*.0005 if sell else 0.)

def roundtrip(t,bid,ask,bv,av,signal,threshold,direction,horizon,start,end,budget=100000):
 """Sell settled opening inventory, repurchase once. No resale of today's purchases."""
 candidates=np.flatnonzero((t>=start)&(t<end-horizon-6)&(direction*signal>=threshold))
 for i in candidates:
  j=np.searchsorted(t,t[i]+1)
  if j>=len(t) or t[j]-t[i]>6: continue
  k=np.searchsorted(t,t[j]+horizon)
  if k>=len(t) or t[k]>end or t[k]-(t[j]+horizon)>6: continue
  # Preallocated fixed inventory at session start; order uses signal-time displayed capacity.
  qty=int(min(10000/bid[0],.01*bv[i])/100)*100
  if qty<100 or qty>.1*bv[j]: continue
  sp=round(float(bid[j])-.01,2); bp=round(float(ask[k])+.01,2)
  # A failed buy-back would leave inventory risk; mark unfilled as failure, never quietly drop it.
  exit_capacity=qty<=.1*av[k]
  gross=qty*(sp-bp); costs=fee(qty*sp,True)+fee(qty*bp)
  return dict(signal_time=float(t[i]),sell_time=float(t[j]),buy_time=float(t[k]),quantity=qty,sell_price=sp,buy_price=bp,gross_cny=gross,cost_cny=costs,net_cny=gross-costs,net_bps=(gross-costs)/(qty*sp)*1e4,exit_capacity=bool(exit_capacity),notional=qty*sp)
 return None

def boot(values):
 a=np.asarray(values,float)
 if len(a)<2:return [None,None]
 rng=np.random.default_rng(42)
 return np.quantile(rng.choice(a,(5000,len(a)),replace=True).mean(1),[.025,.975]).tolist()

def main(root):
 paths={k:root/f'{k}_20260923.parquet' for k in ['snp','exe','ord']}
 inventory=[]; raw={}
 for k,p in paths.items():
  d=pl.read_parquet(p);raw[k]=d
  inventory.append(dict(file=p.name,rows=d.height,symbols=d['Symbol'].n_unique(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),schema={c:str(v) for c,v in d.schema.items()}))
 snp=raw['snp']; t=seconds(snp['Time']); bid=snp['BidPrice1'].to_numpy();ask=snp['AskPrice1'].to_numpy()
 valid=session(t)&(bid>0)&(ask>bid)&(snp['BidVolume1'].to_numpy()>0)&(snp['AskVolume1'].to_numpy()>0)
 duplicate=snp.select(pl.struct(['Symbol','Time']).is_duplicated().sum()).item()
 s=snp.filter(pl.Series(valid)).sort(['Symbol','Time']).unique(['Symbol','Time'],keep='last',maintain_order=True)
 quality=dict(raw_snapshots=snp.height,valid_continuous_snapshots=s.height,removed=snp.height-s.height,duplicate_symbol_time_rows=duplicate,locked_or_crossed=int(np.sum(session(t)&(bid>0)&(ask<=bid))),zero_quote=int(np.sum(session(t)&((bid<=0)|(ask<=0)))))
 exe=raw['exe'];et=seconds(exe['Time']); e=exe.filter(pl.Series(session(et)&(exe['Price'].to_numpy()>0)&(exe['Volume'].to_numpy()>0))).with_columns(pl.Series('minute',np.floor(et[session(et)&(exe['Price'].to_numpy()>0)&(exe['Volume'].to_numpy()>0)]/60).astype(int)))
 em=e.group_by(['Symbol','minute']).agg(pl.col('Volume').sum().alias('volume'),(pl.col('Volume')*(pl.col('BSFlag')==66)).sum().alias('buy'),(pl.col('Volume')*(pl.col('BSFlag')==83)).sum().alias('sell'),(pl.col('Price')*pl.col('Volume')).sum().alias('turnover'))
 # Unknown flags excluded from signed denominator. Full preceding completed minute only.
 flows={(r['Symbol'],r['minute']):(r['buy']-r['sell'])/(r['buy']+r['sell']) if r['buy']+r['sell'] else 0 for r in em.to_dicts()}
 groups={};processed=[]
 for key,g in s.group_by('Symbol',maintain_order=True):
  symbol=int(key[0]); tt=seconds(g['Time']); b=g['BidPrice1'].to_numpy().astype(float);a=g['AskPrice1'].to_numpy().astype(float)
  bvol=g['BidVolume1'].to_numpy();avol=g['AskVolume1'].to_numpy();mid=(a+b)/2
  obi=(bvol-avol)/(bvol+avol)
  b5=g.select(pl.sum_horizontal([f'BidVolume{i}' for i in range(1,6)])).to_series().to_numpy();a5=g.select(pl.sum_horizontal([f'AskVolume{i}' for i in range(1,6)])).to_series().to_numpy()
  obi5=(b5-a5)/(b5+a5)
  ti=np.array([flows.get((symbol,int(x//60)-1),0.) for x in tt])
  groups[symbol]=dict(t=tt,bid=b,ask=a,bv=bvol,av=avol,signals={'obi1':obi,'obi5':obi5,'trade_imbalance':ti},mid=mid)
  processed.append(pl.DataFrame({'symbol':symbol,'time_seconds':tt,'mid':mid,'spread_bps':(a-b)/mid*1e4,'obi1':obi,'obi5':obi5,'trade_imbalance':ti}))
 features=pl.concat(processed);features.write_parquet(OUT/'features.parquet')
 em.write_csv(OUT/'minute_trades.csv')
 # Explore predictability separately from actionable backtest; overlapping labels are descriptive only.
 curves=[]
 for h in [30,60,300,900]:
  for symbol,g in groups.items():
   tt=g['t']; idx=np.searchsorted(tt,tt+h); good=idx<len(tt); ii=np.flatnonzero(good);kk=idx[good]
   good2=(tt[kk]-tt[ii]-h<=6)&(((tt[ii]<41400)&(tt[kk]<41400))|((tt[ii]>=46800)&(tt[kk]<53820)))
   ii=ii[good2];kk=kk[good2]
   ret=(g['mid'][kk]/g['mid'][ii]-1)*1e4
   for split,mask in [('morning',tt[ii]<41400),('afternoon',tt[ii]>=46800)]:
    for binid in range(10):
     ob=g['signals']['obi1'][ii];m=mask&(np.minimum(((ob+1)*5).astype(int),9)==binid)
     if m.sum():curves.append(dict(symbol=symbol,horizon=h,split=split,bin=binid,n=int(m.sum()),mean_bps=float(ret[m].mean())))
 pl.DataFrame(curves).write_csv(OUT/'signal_curves.csv')
 train_symbols=[x for x in groups if x%5!=0]; heldout=[x for x in groups if x%5==0]
 candidates=[]
 for feature in ['obi1','obi5','trade_imbalance']:
  for direction in [-1,1]:
   for threshold in [.6,.8]:
    for h in [300,900,1800]:
     trades=[]
     for symbol in train_symbols:
      g=groups[symbol];r=roundtrip(**{k:g[k] for k in ['t','bid','ask','bv','av']},signal=g['signals'][feature],threshold=threshold,direction=direction,horizon=h,start=34500,end=41400)
      if r:trades.append(r)
     executable=[r for r in trades if r['exit_capacity']]
     score=sum(r['net_cny'] for r in trades)/sum(r['notional'] for r in trades)*1e4 if trades else -1e9
     candidates.append(dict(feature=feature,direction=direction,threshold=threshold,horizon=h,trades=len(trades),capacity_failures=len(trades)-len(executable),train_net_bps=score,eligible=len(trades)>=10 and len(executable)==len(trades)))
 ranking=sorted(candidates,key=lambda x:(x['eligible'],x['train_net_bps']),reverse=True)
 selected=ranking[0];pl.DataFrame(ranking).write_csv(OUT/'training_grid.csv')
 alltrades=[]
 for split,start,end in [('morning',34500,41400),('afternoon',47100,53820)]:
  for symbol,g in groups.items():
   r=roundtrip(**{k:g[k] for k in ['t','bid','ask','bv','av']},signal=g['signals'][selected['feature']],threshold=selected['threshold'],direction=selected['direction'],horizon=selected['horizon'],start=start,end=end)
   if r:alltrades.append(dict(symbol=symbol,split=split,heldout=symbol in heldout,**r))
 trades=pl.DataFrame(alltrades);trades.write_csv(OUT/'trades.csv')
 metrics={}
 for name,rs in [('morning_train',[r for r in alltrades if r['split']=='morning' and not r['heldout']]),('afternoon_all',[r for r in alltrades if r['split']=='afternoon']),('afternoon_heldout',[r for r in alltrades if r['split']=='afternoon' and r['heldout']])]:
  n=len(rs);notional=sum(r['notional'] for r in rs);net=sum(r['net_cny'] for r in rs)
  metrics[name]=dict(trades=n,capacity_failures=sum(not r['exit_capacity'] for r in rs),net_cny=net,gross_cny=sum(r['gross_cny'] for r in rs),cost_cny=sum(r['cost_cny'] for r in rs),traded_notional=notional,net_bps=net/notional*1e4 if notional else None,win_rate=sum(r['net_cny']>0 for r in rs)/n if n else None,mean_trade_bps_ci95=boot([r['net_bps'] for r in rs]),return_on_allocated_inventory_bps=net/(len(groups)*100000)*1e4)
 afternoon=[r for r in alltrades if r['split']=='afternoon']
 sensitivity=[dict(extra_cost_bps=x,net_cny=sum(r['net_cny']-r['notional']*x/1e4 for r in afternoon)) for x in [0,2,5,10,20]]
 summary=dict(date='2026-09-23',inventory=inventory,quality=quality,selected=selected,train_symbols=train_symbols,heldout_symbols=heldout,metrics=metrics,sensitivity=sensitivity,parameters=dict(budget_per_symbol=100000,latency_seconds=1,slippage_ticks_per_leg=1,entry_participation_top_depth=.01,exit_participation_top_depth=.1,max_order_notional=10000,commission_rate=.0002,commission_min=5,transfer_per_leg=.00001,sell_stamp_tax=.0005,roundtrips_per_symbol_per_session=1),evaluation_status='Exploratory chronological split: sizing revised after pilot afternoon inspection. Failed-capacity P&L is hypothetical markout, not executable cash.',limitations=['Single date only; no independent day validation.','Morning parameter search has selection bias; afternoon is the only time holdout.','Starting settled inventory and cash required; repurchases cannot be resold that day.','Snapshot depth is not an execution guarantee; 1-second latency and 1-tick slippage are assumptions.','No full order-book reconstruction; order codes are summarized without inventing their semantics.','Stock bootstrap does not remove common intraday market shocks.','Returns measure incremental cash against holding identical shares, not total portfolio return.'])
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
 # Descriptive order / execution code distributions, quality and activity.
 codes={}
 for name,d in raw.items():
  codes[name]={c:d.group_by(c).len().sort(c).to_dicts() for c in ['FunctionCode','OrderKind','BSFlag'] if c in d.columns}
 (OUT/'code_distributions.json').write_text(json.dumps(codes,indent=2))
 fig,ax=plt.subplots(2,2,figsize=(12,8))
 activity=em.group_by('minute').agg(pl.col('turnover').sum()).sort('minute');mins=activity['minute'].to_numpy();[ax[0,0].plot(mins[m]/60,activity['turnover'].to_numpy()[m]/1e8,color='#276b92') for m in [mins<690,mins>=780]];ax[0,0].set(xlabel='Hour (China time)',ylabel='Turnover (CNY 100m/min)',title='Continuous-session trading activity')
 spreads=features['spread_bps'].to_numpy();ax[0,1].hist(spreads,bins=np.linspace(0,np.quantile(spreads,.99),50),color='#276b92');ax[0,1].set(xlabel='Spread (bps; clipped at 99th pct)',title='Executable quotes have nonzero spread')
 for h in [30,60,300,900]:
  rows=[r for r in curves if r['horizon']==h and r['split']=='afternoon'];ys=[]
  for b in range(10):
   rr=[r for r in rows if r['bin']==b];ys.append(np.average([r['mean_bps'] for r in rr],weights=[r['n'] for r in rr]) if rr else np.nan)
  ax[1,0].plot(np.arange(10)*.2-.9,ys,marker='o',label=f'{h}s')
 ax[1,0].axhline(0,color='black',lw=.7);ax[1,0].legend();ax[1,0].set(xlabel='L1 queue imbalance bin center',ylabel='Forward mid return (bps)',title='Afternoon descriptive response (overlapping labels)')
 stock=features.group_by('symbol').agg(pl.col('spread_bps').median()).sort('spread_bps');ax[1,1].bar(np.arange(stock.height),stock['spread_bps']);ax[1,1].set(xlabel='Stocks ranked by median spread',ylabel='Median spread (bps)',title='Execution costs vary across stocks')
 fig.tight_layout();fig.savefig(OUT/'microstructure.png');plt.close(fig)
 fig,ax=plt.subplots(1,3,figsize=(15,4))
 rs=sorted(afternoon,key=lambda r:r['symbol']);ax[0].bar([str(r['symbol']) for r in rs],[r['net_cny'] for r in rs],color=['#228866' if r['net_cny']>0 else '#c34d50' for r in rs]);ax[0].tick_params(axis='x',rotation=90,labelsize=7);ax[0].set(ylabel='Incremental CNY vs holding',title='Afternoon exploratory markout: per-stock P&L')
 ordered=sorted(rs,key=lambda r:r['buy_time']);ax[1].step(np.arange(len(ordered)+1),np.r_[0,np.cumsum([r['net_cny'] for r in ordered])],where='post');ax[1].set(xlabel='Completed pairs ordered by buy-back time',ylabel='Incremental CNY',title='Realized cash improvement (not equity curve)')
 ax[2].bar([str(r['extra_cost_bps']) for r in sensitivity],[r['net_cny'] for r in sensitivity]);ax[2].set(xlabel='Additional execution cost (bps)',ylabel='Incremental CNY',title='Cost sensitivity after base fees + slippage')
 fig.tight_layout();fig.savefig(OUT/'strategy.png');plt.close(fig)
 print(json.dumps({k:summary[k] for k in ['quality','selected','metrics','sensitivity']},indent=2))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=Path('sse50_20260923'));main(parser.parse_args().data)


