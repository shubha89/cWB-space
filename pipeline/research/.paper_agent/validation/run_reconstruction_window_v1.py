"""D028: fixed, observed-only local growth; no source generation or manuscript edits."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,time,json,csv,hashlib,argparse,signal,resource,platform
from pathlib import Path
from collections import deque
from dataclasses import replace
import importlib.metadata as md
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.setdefault('NUMBA_CACHE_DIR',str(ROOT/'.paper_agent/validation/.numba_cache'))
import numpy as np
from wdm_wavelet.wdm import WDM
DAY=86400.;LIMIT=4*1024**3

def check():
 if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>LIMIT:raise MemoryError('4GB peak memory ceiling')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def inv(w,t):return .5*(np.asarray(w.w2t(t).value)+np.asarray(w.w2tQ(t).value))
def metrics(x,y):
 x=np.concatenate(x);y=np.concatenate(y);a=np.linalg.norm(x);b=np.linalg.norm(y)
 if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Nonfinite series')
 return dict(overlap=float(x@y/(a*b)) if a*b else None,norm_ratio=float(b/a) if a else None,relative_l2=float(np.linalg.norm(y-x)/a) if a else None)
def grow(original,eligible):
 result=original.copy();nf,nt=result.shape;q=deque(zip(*np.nonzero(result)))
 offsets=[(-1,0),(1,0)]+[(df,dt) for dt in [-1,1] for df in range(-4,5)]
 while q:
  f,t=q.popleft()
  for df,dt in offsets:
   ff,tt=f+df,t+dt
   if 0<=ff<nf and 0<=tt<nt and eligible[ff,tt] and not result[ff,tt]:result[ff,tt]=True;q.append((ff,tt))
 return result

def select_observed(times,observed,original,grid_times,grid_freqs):
 """Truth is not accepted by this selection function."""
 w=WDM(M=512,K=256,beta_order=6,precision=6,backend='numba');maps={};power=0;floors={};valid=None
 for c,x in observed.items():
  tf=w.t2w(x,sample_rate=.25,t0=float(times[0]));centers=tf.t0+(np.arange(tf.n_time)+.5)*tf.dt;v=centers<=times[-1]
  if not np.array_equal((centers[v]-times[0])/DAY,grid_times) or not np.array_equal(np.arange(tf.n_freq)*tf.df,grid_freqs):raise ValueError('Original/new grid mismatch')
  if valid is not None:assert np.array_equal(valid,v)
  valid=v;maps[c]=tf;raw=np.abs(tf.data[:,valid])**2
  if not np.isfinite(raw).all():raise ValueError('Nonfinite observed power')
  floor=np.median(raw,axis=1)
  if not np.all(np.isfinite(floor)&(floor>0)):raise ValueError('Nonpositive floor: halt rather than changing normalization')
  floors[c]=floor;power=power+np.abs(tf.data[:,valid]/np.sqrt(floor[:,None]))**2
 check();assert original.shape==power.shape
 occupied=np.flatnonzero(original.any(axis=0));first,last=int(occupied[0]),int(occupied[-1]);left=max(0,first-2);right=min(len(grid_times)-1,last+2)
 band=(grid_freqs>=1e-5)&(grid_freqs<=.02);guard=(grid_times>=.5)&(grid_times<=13.5)
 assert not np.any(original[~band]);threshold=float(np.quantile(power[np.ix_(band,guard)],.95))
 window=np.zeros_like(original);window[np.ix_(band,np.arange(left,right+1))]=True
 eligible=(power>=threshold)&window;expanded=grow(original,eligible);assert np.all(expanded[original]);assert not np.any((expanded&~original)&~eligible)
 fullmask=np.zeros(maps['A2'].data.shape,bool);fullmask[:,valid]=expanded
 added=expanded&~original;disconnected=eligible&~expanded
 info=dict(original_pixel_count=int(original.sum()),expanded_pixel_count=int(expanded.sum()),added_pixel_count=int(added.sum()),threshold=threshold,old_center_interval_days=[float(grid_times[first]),float(grid_times[last])],reconstruction_window_days=[float(grid_times[left]),float(grid_times[right])],margin_bins=2,margin_seconds=float(2*tf.dt),band_hz=[1e-5,.02],growth_quantile=.95,threshold_domain_days=[.5,13.5],normalization='K256 per-channel median over all valid full-record centers',window_boundary_contact=dict(left=bool(expanded[:,left].any()),right=bool(expanded[:,right].any()),top_frequency=bool(expanded[np.flatnonzero(band)[-1]].any())),disconnected_eligible_count=int(disconnected.sum()),added_normalized_power=float(power[added].sum()),disconnected_normalized_power=float(power[disconnected].sum()))
 return w,maps,valid,expanded,fullmask,power,floors,info

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();out=args.output.resolve()
 if out.exists() or out!=ROOT/'.paper_agent/validation/reconstruction_window_v1':raise ValueError('Require fresh approved output path')
 out.mkdir();started=time.monotonic();signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('30minute ceiling')));signal.alarm(1800)
 report=dict(status='RUNNING',command=sys.argv,script_sha256=sha(Path(__file__)),python=sys.version,platform=platform.platform(),packages={x:md.version(x) for x in ['numpy','wdm-wavelet','numba']},selection='Observed data and original rank1 masks only; 2bin extension; unchangedband; fixed95percentile connectedgrowth',cases=[])
 try:
  for n,rel in enumerate(['precessing_v1/case01','precessing_v1/case02','precessing_earlier_start_v1/case03'],1):
   base=ROOT/'.paper_agent/validation'/rel;dest=out/f'case{n:02}';dest.mkdir();row=next(csv.DictReader((base/'all_candidates.csv').open()));assert int(row['rank'])==1
   src=base/'reconstruction.npz';d=np.load(src);times=d['times'];obs={c:d[c+'_observed'] for c in ['A2','E2']};original=np.load(base/'discovered_masks.npz')[row['event_id']].astype(bool)
   gridt=d['wdm_times_day'];gridf=d['wdm_freqs_hz'];w,maps,valid,expanded,fullmask,power,floors,info=select_observed(times,obs,original,gridt,gridf)
   np.savez_compressed(dest/'selection.npz',original_mask=original,expanded_mask=expanded,normalized_power=power,grid_times_day=gridt,grid_freqs_hz=gridf,**{c+'_floor':x for c,x in floors.items()})
   # Freeze observed-only selection before loading truth or computing comparison metrics.
   info.update(case=n,event_id=row['event_id'],original_ranking_statistic=float(row['event_statistic']),selection_sha256=sha(dest/'selection.npz'),input_hashes={str(p.relative_to(ROOT)):sha(p) for p in [src,base/'all_candidates.csv',base/'discovered_masks.npz']})
   (dest/'selection.json').write_text(json.dumps(info,indent=2));new={};old={};controls={}
   oldfull=np.zeros(fullmask.shape,bool);oldfull[:,valid]=original
   for c,tf in maps.items():
    new[c]=inv(w,replace(tf,data=np.where(fullmask,tf.data,0j)));old[c]=d['rank1_'+c+'_reconstruction'];baseline=inv(w,replace(tf,data=np.where(oldfull,tf.data,0j)))
    if not np.allclose(baseline,old[c],rtol=1e-10,atol=1e-18):raise ValueError('Old inverse reproduction mismatch')
    controls[c]=dict(baseline_max_abs_difference=float(np.max(np.abs(baseline-old[c]))),full_map=metrics([obs[c]],[inv(w,tf)]))
   truth={c:d[c+'_source'] for c in ['A2','E2']};rel=(times-times[0])/DAY;comparisons={}
   for domain,edges in [('original_interval',info['old_center_interval_days']),('expanded_window',info['reconstruction_window_days']),('full_record',[float(rel[0]),float(rel[-1])])]:
    mask=(rel>=edges[0])&(rel<=edges[1]);comparisons[domain]=dict(interval_days=edges,n_samples=int(mask.sum()),old=metrics([x[mask] for x in truth.values()],[x[mask] for x in old.values()]),new=metrics([x[mask] for x in truth.values()],[x[mask] for x in new.values()]))
   info.update(comparisons=comparisons,controls=controls);np.savez_compressed(dest/'waveforms.npz',times=times,**{c+'_source':x for c,x in truth.items()},**{c+'_old':x for c,x in old.items()},**{c+'_new':x for c,x in new.items()})
   info['waveforms_sha256']=sha(dest/'waveforms.npz');(dest/'report.json').write_text(json.dumps(info,indent=2));report['cases'].append(info);check();print(json.dumps({'case':n,'pixels':[info['original_pixel_count'],info['expanded_pixel_count']],'expanded_window':comparisons['expanded_window']}),flush=True)
   (out/'report.json').write_text(json.dumps(report,indent=2));d.close()
  report['status']='COMPLETED';signal.alarm(0)
 except Exception as e:
  report['status']='BLOCKED';report['error']=repr(e);raise
 finally:
  report['elapsed_s']=time.monotonic()-started;report['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;(out/'report.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
