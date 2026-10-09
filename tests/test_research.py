import numpy as np
import polars as pl
from research import seconds,session,fee,roundtrip
from execution_core import depth_buy

def test_timestamp_milliseconds():
 assert seconds([93000123])[0]==34200.123

def test_lunch_and_closing_auction_excluded():
 assert session(np.array([34200,41399,41400,45000,46800,53820])).tolist()==[True,True,False,False,True,False]

def test_fee_minimum_and_stamp_tax():
 assert fee(1000)==5.01
 assert fee(1000,True)==5.51
 assert abs(fee(100000,True)-71)<1e-10

def test_roundtrip_latency_lot_and_no_repeat():
 t=np.arange(34500,36001,3.);n=len(t)
 r=roundtrip(t,np.ones(n)*10,np.ones(n)*10.01,np.ones(n)*100000,np.ones(n)*100000,np.ones(n)*-1,.6,-1,300,34500,36000)
 assert r['sell_time']>r['signal_time']
 assert r['buy_time']>=r['sell_time']+300
 assert r['quantity']%100==0
 assert r['net_cny']<0

def test_no_forward_fill_across_lunch():
 t=np.array([41390.,46800.,47700.]);ones=np.ones(3)
 assert roundtrip(t,ones*10,ones*10.01,ones*100000,ones*100000,-ones,.6,-1,900,34500,41400) is None

def test_depth_walk_and_capacity_failure():
 row={}
 for k in range(1,11):
  row[f'AskPrice{k}']=[10+k*.01];row[f'AskVolume{k}']=[1000.]
 g=pl.DataFrame(row)
 price,ok=depth_buy(g,np.int64(0),200)
 assert ok and abs(price-10.025)<1e-9
 assert depth_buy(g,0,1100)[1] is False
