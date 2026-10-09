"""Audit event activity without assuming undocumented order-code semantics."""
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from research import seconds,session,OUT
frames=[]
for kind in ['ord','exe']:
 d=pl.read_parquet(f'sse50_20260923/{kind}_20260923.parquet');t=seconds(d['Time']);mask=session(t)
 d=d.filter(pl.Series(mask)).with_columns(pl.Series('minute',np.floor(t[mask]/60).astype(int)))
 cols=['Symbol','minute']+(['OrderKind'] if kind=='ord' else [])
 q=d.group_by(cols).agg(pl.len().alias('events'),pl.col('Volume').sum().alias('volume'));q.write_csv(OUT/f'{kind}_minute_activity.csv')
 frames.append(q)
fig,ax=plt.subplots(1,2,figsize=(12,4))
for code,label in [(65,'OrderKind A (65)'),(68,'OrderKind D (68)')]:
 q=frames[0].filter(pl.col('OrderKind')==code).group_by('minute').agg(pl.col('events').sum()).sort('minute');t=q['minute'].to_numpy()
 for m in [t<690,t>=780]:ax[0].plot(t[m]/60,q['events'].to_numpy()[m],label=label if np.any(t[m]<690) else None)
ax[0].set(xlabel='Hour (China time)',ylabel='Events per minute',title='Order event codes: descriptive activity');ax[0].legend()
q=frames[1].group_by('minute').agg(pl.col('events').sum()).sort('minute');t=q['minute'].to_numpy()
for m in [t<690,t>=780]:ax[1].plot(t[m]/60,q['events'].to_numpy()[m],color='#236d86')
ax[1].set(xlabel='Hour (China time)',ylabel='Trades per minute',title='Execution event intensity');fig.tight_layout();fig.savefig(OUT/'order_activity.png');plt.close(fig)
