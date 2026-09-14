"""Replot paper Figures 7, 8 and 10 from lossless displayed vectors.

No waveform generation, clustering, reconstruction, or statistical analysis.
Full-domain metrics and their domain qualifications are in data/reconstruction/metrics.json.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "reconstruction"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
args = parser.parse_args()
OUT = args.output_dir
OUT.mkdir(parents=True, exist_ok=True)
meta = json.loads((DATA / "plot_metadata.json").read_text())
plt.rcdefaults()

# Figure 7: original raw observed-cluster reconstruction, no fitted rescaling.
data=np.load(DATA/'figure7.npz')
fig,axs=plt.subplots(3,2,figsize=(8.3,7.4),layout='constrained')
for i,record in enumerate(meta['figure7']):
    src=record['id'];days=data[src+'_days']
    for j,c in enumerate(['A2','E2']):
        ax=axs[i,j];a=data[f'{src}_{c}_source'];b=data[f'{src}_{c}_reconstruction']
        ax.plot(days,a,color='#202a35',lw=1,label='Injected source')
        ax.plot(days,b,color='#c54e22',lw=.9,ls='--',label='Observed-cluster reconstruction')
        ax.set_title(f'MBHB {i+1}, {c}',fontsize=10)
        ax.set_ylabel(f'{c} [Hz]');ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useMathText=True)
        ax.ticklabel_format(axis='x',style='plain',useOffset=False);ax.grid(alpha=.15)
        ax.tick_params(labelsize=8)
        if i==2:ax.set_xlabel('Time from data start [days]')
handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=2,frameon=False,fontsize=9)
for ext in ['png','pdf']:fig.savefig(OUT/f'figure7_observed_reconstruction.{ext}',dpi=180)
plt.close(fig)

# Figure 8: full-record envelope and exact samples in the expanded windows.
plt.rcParams.update({"font.family":"serif","mathtext.fontset":"stix","pdf.fonttype":42})
data=np.load(DATA/'figure8.npz')
fig,axs=plt.subplots(3,3,figsize=(9.2,7.3),layout='constrained',gridspec_kw={'width_ratios':[.9,1,1]})
for i,record in enumerate(meta['figure8']):
    src=record['id'];times=data[src+'_days'];lo,hi=record['window_days'];oldlo,oldhi=record['original_window_days'];mass=record['mass']
    exponent=int(np.floor(np.log10(mass)));mantissa=mass/10**exponent;masslabel=rf'10^{{{exponent}}}' if mantissa==1 else rf'{mantissa:g}\times10^{{{exponent}}}'
    ax=axs[i,0];ax.semilogy(data[src+'_envelope_days'],data[src+'_envelope_max'],color='#202a35',lw=1.);ax.axvspan(lo,hi,color='#bd491f',alpha=.28)
    ax.set_title(rf'$M_{{\rm det}}={masslabel}\,M_\odot$',fontsize=10);ax.set_ylabel('Source amplitude [Hz]',fontsize=9);ax.set_xlim(0,14)
    for j,c in enumerate(['A2','E2'],1):
        ax=axs[i,j];ax.plot(times,data[src+'_'+c+'_source'],color='#202a35',lw=1.05,label='Injected TDI contribution')
        ax.plot(times,data[src+'_'+c+'_new'],color='#bd491f',ls='--',lw=.95,label='Expanded-region reconstruction')
        for edge in [oldlo,oldhi]:ax.axvline(edge,color='#688694',ls=':',lw=.8)
        ax.set_title(c[0]+' near leading candidate',fontsize=9);ax.set_ylabel(c[0]+' [Hz]',fontsize=9);ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useMathText=True);ax.ticklabel_format(axis='x',style='plain',useOffset=False)
    for ax in axs[i]:
        ax.tick_params(labelsize=8);ax.grid(alpha=.13)
        if i==2:ax.set_xlabel('Time from record start [days]',fontsize=8)
handles,labels=axs[0,1].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',frameon=False,ncol=2,fontsize=9)
for ext in ['pdf','png']:fig.savefig(OUT/f'figure8_precessing_reconstruction.{ext}',dpi=190,bbox_inches='tight',pad_inches=.08)
plt.close(fig)

# Figure 10: aggregate source-only TDI versus expanded-region reconstruction.
data=np.load(DATA/'figure10.npz')
fig,axs=plt.subplots(6,2,figsize=(8.2,9.3),layout='constrained')
for i,record in enumerate(meta['figure10']):
    src=record['id'];rank=record['rank'];days=data[src+'_days'];tc=record['catalog_coalescence_day'];span=(tc-.012,tc+.012);lo,hi=record['original_window_days']
    for j,c in enumerate(['A','E']):
        ax=axs[i,j];ax.plot(days,data[src+'_'+c+'_source'],color='#263544',lw=1.15,label='Aggregate injected MBHB contribution')
        ax.plot(days,data[src+'_'+c+'_new'],color='#D55E00',lw=1.05,ls='--',label='Expanded-region reconstruction')
        for edge in [lo,hi]:
            if span[0]<edge<span[1]:ax.axvline(edge,color='#527f91',ls=':',lw=.9)
        ax.axvline(tc,color='#777777',lw=.6,alpha=.6)
        ax.set_title(f'Rank {rank}, {c}',fontsize=10);ax.set_ylabel('')
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useMathText=True);ax.ticklabel_format(axis='x',useOffset=False,style='plain');ax.tick_params(labelsize=8.5);ax.xaxis.set_major_locator(MaxNLocator(nbins=3,min_n_ticks=3));ax.grid(axis='y',alpha=.16,linewidth=.6);ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
        if i==5:ax.set_xlabel('Time from data start [days]')
fig.supylabel('Fractional frequency',fontsize=10)
handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',frameon=False,fontsize=8,ncol=2)
for ext in ['pdf','png']:fig.savefig(OUT/f'figure10_sangria_reconstruction.{ext}',dpi=180,bbox_inches='tight',pad_inches=.08)
plt.close(fig)
print(f'Figures 7, 8 and 10 written to {OUT}')
