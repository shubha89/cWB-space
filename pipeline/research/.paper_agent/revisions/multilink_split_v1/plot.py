from pathlib import Path
import os,json
R=Path(__file__).resolve().parent;ROOT=R.parents[2];P=ROOT/'.paper_agent/validation/clean_glitch_suite_v1';os.environ.setdefault('MPLCONFIGDIR',str(P.parent/'.mplconfig'))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
fig.savefig(ROOT/'cWB_Space_over/figures/multilink_current.pdf');fig.savefig(R/'figure.png',dpi=170);(R/'points.json').write_text(json.dumps(out,indent=2))
