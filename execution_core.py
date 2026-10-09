def depth_buy(g,index,quantity):
 index=int(index);remaining=quantity;cost=0.
 for level in range(1,11):
  price=float(g[f'AskPrice{level}'][index]);volume=float(g[f'AskVolume{level}'][index])
  if price<=0 or volume<=0:continue
  take=min(remaining,int(.1*volume));cost+=take*price;remaining-=take
  if remaining==0:return cost/quantity+.01,True
 return float(g['AskPrice1'][index])+.01,False

