from pathlib import Path
import os,json,hashlib
R=Path(__file__).resolve().parent;ROOT=R.parents[2];P=ROOT/'gpt6_astra_validation_scripts/validation/clean_glitch_suite_v1';os.environ.setdefault('MPLCONFIGDIR',str(P.parent/'.mplconfig'))
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
 fig.savefig(ROOT/'cWB_Space_over/figures'/(name+'.pdf'));fig.savefig(R/(name+'.png'),dpi=170);plt.close(fig)
(R/'checks.json').write_text(json.dumps(checks,indent=2))
