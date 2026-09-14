"""Authorized targeted K256 synthesis on unchanged K32 Sangria candidate masks."""
import os,sys,csv,json,time,hashlib,argparse,platform,importlib.metadata as md
from pathlib import Path
from dataclasses import replace,asdict
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.setdefault('NUMBA_CACHE_DIR',str(ROOT/'.paper_agent/validation/.numba_cache'))
import numpy as np
import h5py
from wdm_wavelet.wdm import WDM
from scripts.blind_event_clustering import compute_self_ae_median_wdm,find_seeded_chirp_events,SeededChirpEventConfig
from scripts.clean_wdm import WDMResolution
DAY=86400.
def sha(data):return hashlib.sha256(data).hexdigest()
def ae(d):return {'A':(d['Z']-d['X'])/np.sqrt(2.),'E':(d['X']-2*d['Y']+d['Z'])/np.sqrt(6.)}
def inv(w,tf):return .5*(np.asarray(w.w2t(tf).value)+np.asarray(w.w2tQ(tf).value))
def metrics(x,y):
 if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Invalid series')
 a,b=np.linalg.norm(x),np.linalg.norm(y)
 return dict(relative_l2=float(np.linalg.norm(y-x)/a),norm_ratio=float(b/a),normalized_dot=float(np.dot(x,y)/(a*b)),n_samples=len(x))
def main():
 p=argparse.ArgumentParser();p.add_argument('--chunk',type=int,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 out=args.output.resolve()
 if out.exists() or not out.is_relative_to(ROOT/'.paper_agent/validation/sangria_k256_v1'):raise ValueError('Require new isolated output')
 out.mkdir(parents=True);began=time.monotonic()
 csvpath=ROOT/'data/sangria_full_mbhb_search/sangria_full_mbhb_candidate_events.csv';expected=[r for r in csv.DictReader(csvpath.open()) if int(r['global_rank'])<=6 and int(r['chunk'])==args.chunk]
 if not expected:raise ValueError('No selected candidates')
 count=int(30*DAY/5);first=(args.chunk-1)*count
 with h5py.File(ROOT/'data/mbhb-unblinded.h5','r') as f:data=f['obs/tdi'][first:first+count,0]
 times=data['t'];obs=ae(data);t0=float(times[0]);n=len(times)
 config=SeededChirpEventConfig(low_hz=1e-5,high_hz=.02,seed_quantile=.99,grow_quantile=.95,max_freq_jump_bins=4,same_time_connect_bins=1,min_pixels=8,min_time_bins=2,freq_tolerance_bins=1.,max_events=12)
 maps=compute_self_ae_median_wdm(times,obs['A'],obs['E'],resolution=WDMResolution('dt5_M512',5.,512))
 events,masks=find_seeded_chirp_events(maps,config=config,start_day=.5,stop_day=(times[-1]-times[0])/DAY-.5,label=f'sangria_chunk{args.chunk:02d}')
 audit=[]
 for old in expected:
  found=events[events.event_id==old['event_id']]
  if len(found)!=1:raise ValueError('Event id differs')
  new=found.iloc[0]
  for key in ['n_pixels','n_time_bins','event_statistic','peak_freq_hz','seed_threshold','grow_threshold','ae_total_power','ae_balance','ae_complex_coherence']:
   if not np.isclose(float(new[key]),float(old[key]),rtol=1e-10,atol=1e-12):raise ValueError('Event field mismatch '+key)
  for key in ['start_day','stop_day','peak_time_day']:
   if abs(float(new[key])+t0/DAY-float(old[key]))>1e-10:raise ValueError('Time differs')
  audit.append(dict(global_rank=int(old['global_rank']),event_id=old['event_id'],n_pixels=int(old['n_pixels']),event_statistic=float(old['event_statistic']),status='REPRODUCED'))
 payload={'times':times,'wdm_times_day':maps['times_day']+t0/DAY,'wdm_freqs_hz':maps['freqs_hz']};roundtrips=[];w=WDM(M=512,K=256,beta_order=6,precision=6,backend='numba');grid=[]
 for c,raw in obs.items():
  tf=w.t2w(raw,sample_rate=.2,t0=t0);tf_times=tf.t0+(np.arange(tf.n_time)+.5)*tf.dt;valid=tf_times<=times[-1];freq=np.arange(tf.n_freq)*tf.df
  if not np.array_equal(freq,maps['freqs_hz']) or not np.array_equal((tf_times[valid]-t0)/DAY,maps['times_day']):raise ValueError('K32/K256 grids differ')
  grid.append(dict(channel=c,identical=True,n_freq=tf.n_freq,n_time=tf.n_time,df=float(tf.df),dt=float(tf.dt),filter_half_length=int(w.m_H)))
  full=inv(w,tf);r=metrics(raw,full);roundtrips.append(dict(input=c+'_observed',**r));payload[c+'_observed']=raw;payload[c+'_full_inverse']=full
  for old in expected:
   rank=old['global_rank'];mask=np.zeros(tf.data.shape,dtype=bool);mask[:,valid]=masks[old['event_id']]
   payload['rank'+rank+'_mask']=mask;payload['rank'+rank+'_'+c+'_reconstruction']=inv(w,replace(tf,data=np.where(mask,tf.data,0j)))
 # Source truth only after masks and observed reconstructions are fixed.
 with h5py.File(ROOT/'data/mbhb-unblinded.h5','r') as f:truth=f['sky/mbhb/tdi'][first:first+count,0]
 if not np.array_equal(truth['t'],times):raise ValueError('Truth time differs')
 source=ae(truth)
 for c,raw in source.items():
  tf=w.t2w(raw,sample_rate=.2,t0=t0);full=inv(w,tf);payload[c+'_source']=raw;payload[c+'_source_full_inverse']=full;roundtrips.append(dict(input=c+'_source',**metrics(raw,full)))
 comparisons=[]
 for old in expected:
  rank=old['global_rank'];lo,hi=float(old['start_day']),float(old['stop_day']);local=(times/DAY>=lo)&(times/DAY<=hi)
  payload['rank'+rank+'_comparison_domain']=local
  for domain,select in [('full_chunk_aggregate_not_per_event',np.ones(n,dtype=bool)),('saved_candidate_center_extent',local)]:
   x=np.concatenate([source[c][select] for c in ['A','E']]);y=np.concatenate([payload['rank'+rank+'_'+c+'_reconstruction'][select] for c in ['A','E']])
   comparisons.append(dict(rank=int(rank),channel_combination='AE',domain=domain,start_day=float(times[select][0]/DAY),stop_day=float(times[select][-1]/DAY),**metrics(x,y)))
 report=dict(status='TARGETED_RECONSTRUCTION_DIAGNOSTIC',chunk=args.chunk,events=audit,grid_checks=grid,discovery_config=asdict(config),discovery_K=32,synthesis_config=dict(M=512,K=256,beta_order=6,precision=6,backend='numba',inverse='0.5*(w2t+w2tQ)'),roundtrips=roundtrips,comparisons=comparisons,units='dimensionless fractional-frequency TDI',truth_use='post-mask aggregate source only; no fitting; not isolated individual source',mask_use='original K32 event mask transferred only on exactly matching grids; no dilation or K256 clustering',script_sha256=sha(Path(__file__).read_bytes()),candidate_csv_sha256=sha(csvpath.read_bytes()),observed_slice_sha256=sha(data.tobytes()),source_slice_sha256=sha(truth.tobytes()),packages={k:md.version(k) for k in ['numpy','scipy','pandas','wdm-wavelet','numba','jax','h5py']},python=sys.version,platform=platform.platform(),command=sys.argv,elapsed_s=time.monotonic()-began)
 np.savez_compressed(out/'reconstruction.npz',**payload);(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'roundtrips':roundtrips,'comparisons':comparisons,'seconds':time.monotonic()-began},indent=2),flush=True)
if __name__=='__main__':main()
