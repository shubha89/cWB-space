from pathlib import Path
import sys,os,json,time,hashlib,resource
P=Path(__file__).resolve().parent;ROOT=P.parents[2];sys.path.insert(0,str(ROOT));os.environ.setdefault('MPLCONFIGDIR',str(P.parent/'.mplconfig'));os.environ.setdefault('NUMBA_CACHE_DIR',str(P.parent/'.numba_cache'))
import numpy as np,pandas as pd
import diagnostics.run_generic_burst_response_validation as g
from scripts.clean_wdm import WDMResolution
from scripts.blind_event_clustering import compute_self_ae_median_wdm,find_seeded_chirp_events,SeededChirpEventConfig
start=time.monotonic();cases=json.loads((P/'manifest.json').read_text());assert len(cases)==24 and sum(c['kind']=='glitch' for c in cases)==17;Npath=P.parent/'continuous_noise_control_v1/noise_continuous.npz';noise=np.load(Npath);nm=np.load(P.parent/'continuous_noise_control_v1/continuous/noise_maps.npz');t=noise['t_lisa'];chs=['A2','E2','T2',*g.SAGNAC_CHANNELS];tt=nm['times_day'];ff=nm['frequencies_hz'];domain=((ff>=1e-5)&(ff<=.02))[:,None]&((tt>=4.25)&(tt<=5.75))[None,:];target=1053584.6062279106
_,WDM=g.require_wdm_dependencies();wdm=WDM(M=512,K=32,beta_order=6,precision=6,backend='numba');cfg=SeededChirpEventConfig(low_hz=1e-5,high_hz=.02,seed_quantile=.99,grow_quantile=.95,max_freq_jump_bins=4,same_time_connect_bins=1,min_pixels=8,min_time_bins=2,freq_tolerance_bins=1.,max_events=30)
def power(x):
 c,at,af,_,_=g.compute_wdm_transform(wdm,x,sample_rate=.25,t0=float(t[0]),original_t1=float(t[-1]));assert np.array_equal((at-t[0])/86400+4,tt) and np.array_equal(af,ff);assert np.isfinite(c).all();return abs(c)**2
records=[];rows=[];sumrows=[];hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [Npath,P/'manifest.json',P.parent/'continuous_noise_control_v1/continuous/noise_maps.npz']}
for case in cases:
 cid=case['case_id'];path=Path(case['source_path']);hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();src=np.load(path);assert np.array_equal(src['t_lisa'],t);sq={c:power(src[c]) for c in ['A2','E2']};qnative=sum(float((sq[c]/nm[c+'_floor'][:,None])[domain].sum()) for c in sq);assert qnative>0 and np.isfinite(qnative)
 levels=['native','matched'] if case['kind']=='glitch' else ['native']
 for level in levels:
  scale=1. if level=='native' else float(np.sqrt(target/qnative));rid=cid+'__'+level;O=P/'results'/rid;O.mkdir(parents=True,exist_ok=False);obs={c:noise[c]+scale*src[c] for c in chs};maps=compute_self_ae_median_wdm(t,obs['A2'],obs['E2'],resolution=WDMResolution('dt4_M512',4.,512));maps['times_day']+=4;ev,masks=find_seeded_chirp_events(maps,config=cfg,start_day=4.25,stop_day=5.75,label=rid);np.savez_compressed(O/'search_maps.npz',**maps);np.savez_compressed(O/'masks.npz',**masks)
  qcheck=sum(float((power(scale*src[c])/nm[c+'_floor'][:,None])[domain].sum()) for c in ['A2','E2']);np.testing.assert_allclose(qcheck,qnative*scale**2,rtol=1e-12);np.savez_compressed(O/'strength.npz',domain=domain,**{c+'_power':sq[c]*scale**2 for c in sq})
  rec=dict(record=rid,case_id=cid,group=case['group'],kind=case['kind'],level=level,amplitude_scale=scale,Q=qcheck,Q_native=qnative,n_candidates=len(ev),ae_peak_day=case['ae_peak_day'],coverage=case['ae_timepower_fraction_two_days']);records.append(rec)
  if not len(ev):
   (O/'NO_CANDIDATES.json').write_text(json.dumps(rec,indent=2));print(rid,'NO CANDIDATES',flush=True);continue
  ev.to_csv(O/'search.csv',index=False);union=np.logical_or.reduce(list(masks.values()));used=union.any(axis=1);u=union[used];counts=(~u).sum(axis=1);assert np.all(counts>0);evidence=dict(union_mask=u,used_frequency_indices=np.flatnonzero(used),off_counts=counts,times_day=tt,frequencies_hz=ff);cs=[]
  for c in g.SAGNAC_CHANNELS:
   q=power(obs[c])[used];n=nm[c+'_power'][used];nf=nm[c+'_floor'][used];off=np.array([np.median(v[~mm]) for v,mm in zip(q,u)]);assert np.all(off>0) and np.isfinite(off).all();vals=dict(off_subtracted=np.maximum(q/off[:,None]-1,0),matched_noise=np.maximum((q-n)/nf[:,None],0))
   for k,v in [('observed_power',q),('noise_power',n),('noise_floor',nf),('off_floor',off)]:evidence[c+'_'+k]=v
   for e in ev.event_id:
    mask=masks[e][used];evidence[e+'_mask']=mask;cs.append(dict(record=rid,event_id=e,channel=c,**{k:float(v[mask].sum()) for k,v in vals.items()}))
  np.savez_compressed(O/'evidence.npz',**evidence);sumrows.extend(cs)
  for rank,r in enumerate(ev.to_dict('records'),1):
   r.update(rec);r['rank']=rank;r['associated']=bool(any(r['stop_day']>=a and r['start_day']<=b for a,b in case['extra']['input_intervals'])) if case['kind']=='glitch' else bool(r['start_day']<=case['ae_peak_day']<=r['stop_day']);sub=[s for s in cs if s['event_id']==r['event_id']]
   for mode in vals:
    nn=sum(s[mode] for s in sub if s['channel'] in g.ZETA_CHANNELS);dd=sum(s[mode] for s in sub if s['channel'] in g.ABC_CHANNELS);r[mode+'_N']=nn;r[mode+'_D']=dd;r[mode+'_R']=nn/dd if dd>0 else np.nan;r[mode+'_status']='OK' if dd>0 else 'ZERO_DENOMINATOR'
   rows.append(r)
  print(rid,len(ev),'Q',qcheck,flush=True)
 assert time.monotonic()-start<600 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<8*1024**3
pd.DataFrame(records).to_csv(P/'records.csv',index=False);pd.DataFrame(rows).to_csv(P/'all_candidates.csv',index=False);pd.DataFrame(sumrows).to_csv(P/'channel_sums.csv',index=False);assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items());(P/'analysis_execution.json').write_text(json.dumps(dict(elapsed_s=time.monotonic()-start,maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,input_hashes=hashes,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
