"""Unfitted targeted reconstructions on original timestamps; truth used only for display."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
plt.rcParams.update({"font.family":"serif","mathtext.fontset":"stix","pdf.fonttype":42})
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'.paper_agent/validation/sangria_k256_v1';OUT=ROOT/'.paper_agent/figures/window_v1';OUT.mkdir(parents=True,exist_ok=True)
rows=list(csv.DictReader((ROOT/'.paper_agent/validation/sangria_top_v1/top_candidate_associations.csv').open()))
chunks={3:None,8:None,9:None,11:None};cases={};hashes={}
for chunk in chunks:
 path=BASE/f'chunk{chunk:02d}/reconstruction.npz';chunks[chunk]=np.load(path);hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
 for event in json.loads((path.parent/'report.json').read_text())['events']:cases[event['global_rank']]=chunk
for group,ranks in enumerate([[1,2,3,4,5,6]],1):
 fig,axs=plt.subplots(6,2,figsize=(8.2,9.3),layout='constrained')
 for i,rank in enumerate(ranks):
  row=next(r for r in rows if int(r['global_rank'])==rank);d=np.load(ROOT/f'.paper_agent/validation/sangria_window_v1/rank{rank:02}/waveforms.npz');hashes[str(ROOT/f'.paper_agent/validation/sangria_window_v1/rank{rank:02}/waveforms.npz')]=hashlib.sha256((ROOT/f'.paper_agent/validation/sangria_window_v1/rank{rank:02}/waveforms.npz').read_bytes()).hexdigest();days=d['times']/86400.;tc=float(row['catalog_coalescence_day']);span=(tc-.012,tc+.012);sel=(days>=span[0])&(days<=span[1]);lo=float(row['event_start_day']);hi=float(row['event_stop_day'])
  for j,c in enumerate(['A','E']):
   ax=axs[i,j];ax.plot(days[sel],d[c+'_source'][sel],color='#263544',lw=1.15,label='Aggregate injected MBHB contribution')
   ax.plot(days[sel],d[c+'_new'][sel],color='#D55E00',lw=1.05,ls='--',label='Expanded-region reconstruction')
   for edge in [lo,hi]:
    if span[0]<edge<span[1]:ax.axvline(edge,color='#527f91',ls=':',lw=.9)
   ax.axvline(tc,color='#777777',lw=.6,alpha=.6)
   ax.set_title(f'Rank {rank}, {c}',fontsize=10);ax.set_ylabel('')
   ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useMathText=True);ax.ticklabel_format(axis='x',useOffset=False,style='plain');ax.tick_params(labelsize=8.5);ax.xaxis.set_major_locator(MaxNLocator(nbins=3,min_n_ticks=3));ax.grid(axis='y',alpha=.16,linewidth=.6);ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
   if i==5:ax.set_xlabel('Time from data start [days]')
 fig.supylabel("Fractional frequency",fontsize=10)
 handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',frameon=False,fontsize=8,ncol=2)
 checks=[]
 for ax in fig.axes:
  lines=[[hashlib.sha256(np.asarray(l.get_xdata()).tobytes()).hexdigest(),hashlib.sha256(np.asarray(l.get_ydata()).tobytes()).hexdigest()] for l in ax.lines]
  checks.append(dict(lines=lines,xlim=list(ax.get_xlim()),ylim=list(ax.get_ylim()),xscale=ax.get_xscale(),yscale=ax.get_yscale()))
 for ext in ['pdf','png']:fig.savefig(OUT/f'sangria_targeted_combined.{ext}',dpi=180,bbox_inches='tight',pad_inches=.08)
 plt.close(fig)
(OUT/'sangria_provenance.json').write_text(json.dumps({'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'inputs':hashes,'display':'catalog coalescence +/-0.012day; no alignment shift or fitting; original candidate-center limits dotted if in window; solid vertical line catalog tc'},indent=2)+'\n')
