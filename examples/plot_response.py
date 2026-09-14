#!/usr/bin/env python3
"""Replot Figures 4–6 from saved response maps and candidate summaries.

Run from any directory; --output-dir defaults to ./outputs.
This reproduces saved plotting choices without rerunning the scientific pipeline.
"""
from pathlib import Path
import argparse
import json
import hashlib

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
args = parser.parse_args()
OUTPUT = args.output_dir.resolve()
P = Path(__file__).resolve().parents[1] / "data" / "response"
if OUTPUT == P or P in OUTPUT.parents:
    parser.error("Output directory must be outside the input data directory")
OUTPUT.mkdir(parents=True, exist_ok=True)
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
m=json.loads((P/'manifest.json').read_text());events=pd.read_csv(P/'all_candidates.csv');checks=[]
for group,name in [('primary','primary_reconciled'),('generic','generic_reconciled')]:
 cases=sorted([x for x in m if x['group']==group],key=lambda x:x['kind']=='glitch');fig,axs=plt.subplots(3,len(cases),figsize=(11.3,5.4),sharex=True,sharey=True,layout='constrained');cm=plt.get_cmap('RdBu_r').copy();cm.set_bad('#cccccc');norms=[LogNorm(1,1e4),LogNorm(1,3e3),LogNorm(1e-4,3)]
 for col,case in enumerate(cases):
  cid=case['case_id'];lev='matched' if case['kind']=='glitch' else 'native';d=P/'results'/(cid+'__'+lev);z=np.load(d/'search_maps.npz');ms=np.load(d/'masks.npz');e=events[(events.case_id==cid)&(events.level==lev)];t=z['times_day']-4;f=z['freqs_hz']*1000;fi=(f>=.01)&(f<=20);power=z['AE_power'];np.testing.assert_allclose(power,z['A_power']+z['E_power']);score=np.full(power.shape,np.nan);ratio=score.copy();union=np.zeros(power.shape,bool)
  for _,row in e.iterrows():
   mask=ms[row.event_id];assert not (union&mask).any();union|=mask;score[mask]=row.event_statistic;ratio[mask]=row.off_subtracted_R;assert np.isfinite(row.off_subtracted_R)
  assert int(union.sum())==sum(int(ms[k].sum()) for k in ms.files)
  for i,arr in enumerate([power,score,ratio]):
   im=axs[i,col].pcolormesh(t,f[fi],np.ma.masked_invalid(arr[fi]),cmap=cm,norm=norms[i],shading='nearest',rasterized=True);axs[i,col].set_xlim(0,2);axs[i,col].set_ylim(f[fi][0],20);axs[i,col].set_yscale('log');axs[i,col].set_xticks([0,1,2]);axs[i,col].tick_params(labelsize=7)
   if col==len(cases)-1:
    cb=fig.colorbar(im,ax=axs[i,:],location='right',pad=.012,shrink=.85,extend='both');cb.ax.tick_params(labelsize=7);cb.set_label(['Normalized power',r'$\rho_{\rm event}$',r'$R_\zeta$'][i],fontsize=8)
  p=case['parameters']
  if case['kind']=='mbhb':title={1e5:r'MBHB $10^5$',3e5:r'MBHB $3\times10^5$',1e6:r'MBHB $10^6$'}[p['total_mass']]
  elif case['kind']=='glitch':title=f"Glitch {p['injection_point'][-2:]}\n{p['width_s']:g} s"
  else:title=('SG ' if case['kind']=='sine_gaussian' else 'WNB ')+cid[-2:]
  axs[0,col].set_title(title,fontsize=8);checks.append(dict(case_id=cid,level=lev,candidates=len(e),pixels=int(union.sum()),visible_pixels=int(union[fi].sum()),hashes={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in [d/'search_maps.npz',d/'masks.npz']}));assert union[fi].sum()==union.sum()
 for i,lab in enumerate(['Observed WDM power','Candidate ranking',r'Response ratio $R_\zeta$']):axs[i,0].set_ylabel(lab+'\nFrequency [mHz]',fontsize=8)
 fig.supxlabel('Time within each separate record [days]',fontsize=10)
 fig.savefig(OUTPUT/(name+'.pdf'));fig.savefig(OUTPUT/(name+'.png'),dpi=170);plt.close(fig)
(OUTPUT/'response_checks.json').write_text(json.dumps(checks,indent=2))

from matplotlib.lines import Line2D
m=json.loads((P/'manifest.json').read_text());e=pd.read_csv(P/'all_candidates.csv');e=e[e['rank']==1];cases=[c for c in m if c['kind']=='mbhb' or c['group']=='multilink'];special=lambda c:c['case_id'].startswith(('pair_reciprocal','sixlink'));groups=[[c for c in cases if not special(c)],[c for c in cases if special(c)]]
fig,axs=plt.subplots(2,2,figsize=(10.2,7.1),sharex='col',gridspec_kw={'height_ratios':[13,3.2]},layout='constrained');out=[]
for block,cc in enumerate(groups):
 labels=[]
 for i,c in enumerate(cc):
  if c['kind']=='mbhb':labels.append({1e5:r'MBHB $10^5$',3e5:r'MBHB $3\times10^5$',1e6:r'MBHB $10^6$'}[c['parameters']['total_mass']]);levs=[('native',0,'#236a9b','o')]
  else:
   links=[x['injection_point'][-2:] for x in c['parameters']['components']];lab='+'.join(links) if len(links)<6 else 'All six links';lab+=' (reciprocal)' if c['case_id'].startswith('pair_reciprocal') else '';labels.append(lab);levs=[('native',-.12,'#656565','D'),('matched',.12,'#bd5918','s')]
  for level,dy,color,marker in levs:
   row=e[(e.case_id==c['case_id'])&(e.level==level)].iloc[0];out.append(dict(case_id=c['case_id'],level=level,block=block,event_id=row.event_id,statistic=float(row.event_statistic),R=float(row.off_subtracted_R),timing_check=bool(row.associated)))
   for j,key in enumerate(['event_statistic','off_subtracted_R']):axs[block,j].scatter(row[key],i+dy,s=24,color=color,marker=marker,zorder=3)
 for j,ax in enumerate(axs[block]):
  ax.set_xscale('log');ax.set_yticks(range(len(cc)),labels if j==0 else []);ax.tick_params(labelsize=8);ax.set_ylim(len(cc)-.6,-.6);ax.grid(axis='x',alpha=.2)
  if block==1:ax.set_facecolor('#fff3eb')
 axs[block,0].set_title('(a) Other configurations and GW controls' if block==0 else '(b) Low-ratio glitch configurations',loc='left',fontsize=10)
for j,(lim,title) in enumerate([((5,2500),r'Search statistic $\rho_{\rm event}$'),((5e-4,2),r'Response ratio $R_\zeta$')]):
 axs[1,j].set_xlim(*lim);axs[1,j].set_xlabel(title,fontsize=10)
fig.legend(handles=[Line2D([],[],color='#236a9b',marker='o',ls='',label='GW signals'),Line2D([],[],color='#656565',marker='D',ls='',label='Reference glitches'),Line2D([],[],color='#bd5918',marker='s',ls='',label='Loud glitches')],loc='outside upper center',ncol=3,fontsize=9)
assert len(out)==27 and sum(x['block']==1 for x in out)==4
fig.savefig(OUTPUT/'multilink_current.pdf');fig.savefig(OUTPUT/'multilink_current.png',dpi=170);(OUTPUT/'multilink_points.json').write_text(json.dumps(out,indent=2))

plt.close(fig)
