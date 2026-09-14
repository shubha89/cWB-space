"""D029: extend fixed D028 rule to six saved Sangria candidates, no tuning."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/lisa-paper-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,time,resource,signal
from pathlib import Path
from dataclasses import replace
import numpy as np
from run_reconstruction_window_v1 import WDM,grow,inv,metrics,sha,check,DAY
ROOT=Path(__file__).resolve().parents[2];out=ROOT/'.paper_agent/validation/sangria_window_v1'
if out.exists():raise ValueError('Fresh output required')
out.mkdir();beg=time.monotonic();signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('30minute limit')));signal.alarm(1800)
r=dict(status='RUNNING',script_sha256=sha(Path(__file__)),shared_driver_sha256=sha(Path(__file__).with_name('run_reconstruction_window_v1.py')),command=sys.argv,cases=[])
try:
 for chunk in [3,8,9,11]:
  src=ROOT/f'.paper_agent/validation/sangria_k256_v1/chunk{chunk:02}/reconstruction.npz';d=np.load(src);times=d['times'];t0=float(times[0]);grid=d['wdm_times_day'];freq=d['wdm_freqs_hz'];obs={c:d[c+'_observed'] for c in ['A','E']};w=WDM(M=512,K=256,beta_order=6,precision=6,backend='numba');maps={};power=0;floors={}
  for c,x in obs.items():
   tf=w.t2w(x,sample_rate=.2,t0=t0);centers=tf.t0+(np.arange(tf.n_time)+.5)*tf.dt;valid=centers<=times[-1];relative=(centers[valid]-t0)/DAY
   assert np.array_equal(relative+t0/DAY,grid) and np.array_equal(np.arange(tf.n_freq)*tf.df,freq)
   raw=np.abs(tf.data[:,valid])**2;floor=np.median(raw,axis=1);assert np.isfinite(raw).all() and np.all(floor>0)
   power+=np.abs(tf.data[:,valid]/np.sqrt(floor[:,None]))**2;maps[c]=tf;floors[c]=floor
  band=(freq>=1e-5)&(freq<=.02);guard=(relative>=.5)&(relative<=(times[-1]-t0)/DAY-.5);threshold=float(np.quantile(power[np.ix_(band,guard)],.95));check()
  ranks=sorted(int(k.split('_')[0][4:]) for k in d.files if k.endswith('_mask'))
  for rank in ranks:
   dest=out/f'rank{rank:02}';dest.mkdir();oldfull=d[f'rank{rank}_mask'].astype(bool);original=oldfull[:,valid];occupied=np.flatnonzero(original.any(axis=0));first,last=int(occupied[0]),int(occupied[-1]);left=max(0,first-2);right=min(len(grid)-1,last+2)
   window=np.zeros_like(original);window[np.ix_(band,np.arange(left,right+1))]=True;eligible=(power>=threshold)&window;expanded=grow(original,eligible);full=np.zeros_like(oldfull);full[:,valid]=expanded;assert np.all(expanded[original])
   info=dict(rank=rank,chunk=chunk,original_pixel_count=int(original.sum()),expanded_pixel_count=int(expanded.sum()),threshold=threshold,margin_bins=2,margin_seconds=2*tf.dt,band_hz=[1e-5,.02],threshold_domain_relative_days=[.5,(times[-1]-t0)/DAY-.5],original_interval_days=[float(grid[first]),float(grid[last])],reconstruction_window_days=[float(grid[left]),float(grid[right])],window_boundary_contact=dict(left=bool(expanded[:,left].any()),right=bool(expanded[:,right].any()),top_frequency=bool(expanded[np.flatnonzero(band)[-1]].any())),disconnected_eligible_count=int((eligible&~expanded).sum()),input_path=str(src.relative_to(ROOT)),input_sha256=sha(src))
   np.savez_compressed(dest/'selection.npz',original_mask=original,expanded_mask=expanded,normalized_power=power,grid_times_day=grid,grid_relative_days=relative,grid_freqs_hz=freq,**{c+'_floor':f for c,f in floors.items()});info['selection_sha256']=sha(dest/'selection.npz');(dest/'selection.json').write_text(json.dumps(info,indent=2))
   # Selection serialized before accessing injected aggregate truth.
   old={};new={};controls={}
   for c,tf in maps.items():
    old[c]=d[f'rank{rank}_{c}_reconstruction'];baseline=inv(w,replace(tf,data=np.where(oldfull,tf.data,0j)));assert np.array_equal(baseline,old[c]);new[c]=inv(w,replace(tf,data=np.where(full,tf.data,0j)));controls[c]=dict(baseline_max_abs_difference=float(np.max(np.abs(baseline-old[c]))),full_map=metrics([obs[c]],[inv(w,tf)]))
   truth={c:d[c+'_source'] for c in ['A','E']};comparisons={}
   for domain,edges in [('original_interval',info['original_interval_days']),('expanded_window',info['reconstruction_window_days']),('full_chunk_aggregate_not_individual',[times[0]/DAY,times[-1]/DAY])]:
    m=(times/DAY>=edges[0])&(times/DAY<=edges[1]);comparisons[domain]=dict(interval_days=edges,n_samples=int(m.sum()),old=metrics([x[m] for x in truth.values()],[x[m] for x in old.values()]),new=metrics([x[m] for x in truth.values()],[x[m] for x in new.values()]))
   np.savez_compressed(dest/'waveforms.npz',times=times,**{c+'_source':x for c,x in truth.items()},**{c+'_old':x for c,x in old.items()},**{c+'_new':x for c,x in new.items()});info.update(comparisons=comparisons,controls=controls,waveforms_sha256=sha(dest/'waveforms.npz'));(dest/'report.json').write_text(json.dumps(info,indent=2));r['cases'].append(info);print(json.dumps({'rank':rank,'comparison':comparisons['expanded_window']}),flush=True);check()
  d.close()
 r['status']='COMPLETED'
except Exception as e:r.update(status='BLOCKED',error=repr(e));raise
finally:
 r.update(elapsed_s=time.monotonic()-beg,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,truth='Aggregate injected MBHB channels; not isolated per-event truth',selection='Observed only; original masks plus connected growth; no fit/shift/tuning');(out/'report.json').write_text(json.dumps(r,indent=2))
