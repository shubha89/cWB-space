"""Whole-record context and raw candidate reconstructions from approved runs."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','pdf.fonttype':42})
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'gpt6_astra_validation_scripts/validation/precessing_v1';OUT=ROOT/'gpt6_astra_validation_scripts/figures/window_v1';OUT.mkdir(parents=True,exist_ok=True)
cases=[]
for num in [1,2,3]:
 path=(ROOT/'gpt6_astra_validation_scripts/validation/precessing_earlier_start_v1/case03') if num==3 else BASE/f'case{num:02d}';report=json.loads((path/'report.json').read_text())
 if report['status']!='COMPLETED_TARGETED_EXAMPLE':raise RuntimeError('All three outcomes required')
 cases.append((path,report))
fig,axs=plt.subplots(3,3,figsize=(9.2,7.3),layout='constrained',gridspec_kw={'width_ratios':[.9,1,1]})
provenance={'cases':[],'display':'left: fullrecord512sample/2048s nonoverlapping maximum combinedsource amplitude; right: fixed2bin enlarged window; nofit or shifting','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
for i,(path,report) in enumerate(cases):
 row=next(csv.DictReader((path/'all_candidates.csv').open()));rank=int(row['rank']);assert rank==1
 data=np.load(path/'reconstruction.npz');times=(data['times']-data['times'][0])/86400.
 oldlo=float(row['start_day']);oldhi=float(row['stop_day']);newbase=ROOT/f'gpt6_astra_validation_scripts/validation/reconstruction_window_v1/case{i+1:02}';newdata=np.load(newbase/'waveforms.npz');newreport=json.loads((newbase/'report.json').read_text());lo,hi=newreport['reconstruction_window_days'];sel=(times>=lo)&(times<=hi)
 mass=report['case']['params']['total_mass'];exponent=int(np.floor(np.log10(mass)));mantissa=mass/10**exponent;masslabel=rf'10^{{{exponent}}}' if mantissa==1 else rf'{mantissa:g}\times10^{{{exponent}}}'
 amp=np.hypot(data['A2_source'],data['E2_source']);starts=np.arange(0,len(amp),512);maxamp=np.maximum.reduceat(amp,starts);ends=np.minimum(starts+511,len(amp)-1);centers=(times[starts]+times[ends])/2
 ax=axs[i,0];ax.semilogy(centers,maxamp,color='#202a35',lw=1.);ax.axvspan(lo,hi,color='#bd491f',alpha=.28)
 ax.set_title(rf'$M_{{\rm det}}={masslabel}\,M_\odot$',fontsize=10);ax.set_ylabel('Source amplitude [Hz]',fontsize=9);ax.set_xlim(0,14)
 for j,c in enumerate(['A2','E2'],1):
  ax=axs[i,j];ax.plot(times[sel],data[c+'_source'][sel],color='#202a35',lw=1.05,label='Injected TDI contribution')
  ax.plot(times[sel],newdata[c+'_new'][sel],color='#bd491f',ls='--',lw=.95,label='Expanded-region reconstruction')
  for edge in [oldlo,oldhi]:ax.axvline(edge,color='#688694',ls=':',lw=.8)
  ax.set_title(c[0]+' near leading candidate',fontsize=9);ax.set_ylabel(c[0]+' [Hz]',fontsize=9);ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useMathText=True);ax.ticklabel_format(axis='x',style='plain',useOffset=False)
 for ax in axs[i]:
  ax.tick_params(labelsize=8);ax.grid(alpha=.13)
  if i==2:ax.set_xlabel('Time from record start [days]',fontsize=8)
 provenance['cases'].append({'case':report['case']['id'],'rank':rank,'start_day':lo,'stop_day':hi,'inputs':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [path/'reconstruction.npz',path/'all_candidates.csv',path/'report.json',newbase/'waveforms.npz',newbase/'report.json']}})
handles,labels=axs[0,1].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',frameon=False,ncol=2,fontsize=9)
for ext in ['pdf','png']:fig.savefig(OUT/f'precessing_earlier_start_reconstruction.{ext}',dpi=190,bbox_inches='tight',pad_inches=.08)
plt.close(fig);(OUT/'precessing_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
