"""Display every saved candidate; highlight independently recorded associations."""
from pathlib import Path
import csv, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
root = Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,default=root/'outputs')
args=parser.parse_args()
paths = [root/'data/sangria/candidates.csv', root/'data/sangria/associations.csv']
rows, associations = [list(csv.DictReader(p.open())) for p in paths]
lookup = {r['event_id']: r for r in rows}
assert len(lookup) == len(rows) == 155
assert len(associations) == 6
for a in associations:
    r = lookup[a['event_id']]
    assert int(r['global_rank']) == int(a['global_rank'])
    assert float(r['event_statistic']) == float(a['event_statistic'])
    assert float(r['peak_time_day']) == float(a['event_peak_day'])
ids = {a['event_id'] for a in associations}
others = [r for r in rows if r['event_id'] not in ids]
assert all(np.isfinite(float(r['event_statistic'])) and float(r['event_statistic']) > 0 for r in rows)
plt.rcParams.update({'font.family':'serif','font.size':11,'mathtext.fontset':'stix','axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig, ax = plt.subplots(figsize=(9,3.65), layout='constrained')
ax.scatter([float(r['peak_time_day']) for r in others], [float(r['event_statistic']) for r in others], s=23, color='#939da7', alpha=.85, edgecolors='none', label='Other saved candidates (149)', zorder=3)
ax.scatter([float(a['event_peak_day']) for a in associations], [float(a['event_statistic']) for a in associations], s=66, color='#0072B2', edgecolors='white', linewidth=.7, label='MBHB-associated candidates (6)', zorder=5)
offsets={1:(7,5),2:(7,5),3:(7,5),4:(-15,12),5:(12,1),6:(8,5)}
for a in associations:
    rank=int(a['global_rank'])
    ax.annotate(str(rank), (float(a['event_peak_day']),float(a['event_statistic'])), xytext=offsets[rank], textcoords='offset points', color='#005582', fontsize=11, fontweight='bold', va='bottom', arrowprops=dict(arrowstyle='-',color='#0072B2',lw=.6) if rank in (4,5) else None)
ax.set(xlim=(0,365), yscale='log', ylim=(min(float(r['event_statistic']) for r in rows)*.75,700), xlabel='Time from data start [days]', ylabel=r'Event statistic, $\rho_{\rm event}$')
ax.set_xticks([0,60,120,180,240,300,365])
ax.set_yticks([1,3,10,30,100,300])
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.grid(axis='y', which='major',color='#e5e7eb',lw=.6)
ax.set_axisbelow(True)
ax.legend(loc='lower left', bbox_to_anchor=(0,1.02), ncol=2, frameon=False, fontsize=10, borderaxespad=0)
out=args.output_dir
out.mkdir(parents=True,exist_ok=True)
for ext in ['pdf','png']:
    fig.savefig(out/f'figure9_sangria_candidates.{ext}',dpi=220)
plt.close(fig)
print(f'Figure 9 written to {out}')
