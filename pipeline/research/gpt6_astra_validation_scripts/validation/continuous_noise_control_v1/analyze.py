from pathlib import Path
import sys,os,time,json,hashlib,resource
P=Path(__file__).resolve().parent;ROOT=P.parents[2];sys.path.insert(0,str(ROOT));os.environ.setdefault('NUMBA_CACHE_DIR',str(P.parent/'.numba_cache'));os.environ.setdefault('MPLCONFIGDIR',str(P.parent/'.mplconfig'))
import numpy as np,pandas as pd
import diagnostics.run_multilink_glitch_detection_rzeta_study as l
import diagnostics.run_generic_burst_response_validation as g
from scripts.blind_event_clustering import compute_self_ae_median_wdm,find_seeded_chirp_events,SeededChirpEventConfig
from scripts.clean_wdm import WDMResolution
start=time.monotonic();chs=['A2','E2','T2',*g.SAGNAC_CHANNELS];_,WDM=g.require_wdm_dependencies();wdm=WDM(M=512,K=32,beta_order=6,precision=6,backend='numba');cfg=SeededChirpEventConfig(low_hz=1e-5,high_hz=.02,seed_quantile=.99,grow_quantile=.95,max_freq_jump_bins=4,same_time_connect_bins=1,min_pixels=8,min_time_bins=2,freq_tolerance_bins=1.,max_events=30)
sourcepaths=[P.parent/'half_day_control_v1/sources'/f for f in ['mass_1000000.npz','glitch.npz']];sources=[];hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sourcepaths}
for p in sourcepaths:
 z=np.load(p);idx=slice(int(4*86400/4),int(6*86400/4));sources.append({k:z[k][idx] for k in ['t_lisa',*chs]})
allrows=[];noise_rows=[];strength=[]
for mode in ['continuous','split']:
 O=P/mode;O.mkdir(exist_ok=False);path=P/f'noise_{mode}.npz';hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();noise=np.load(path);t=noise['t_lisa'];assert all(np.array_equal(s['t_lisa'],t) for s in sources)
 def power(x):
  c,tt,ff,_,_=g.compute_wdm_transform(wdm,x,sample_rate=.25,t0=float(t[0]),original_t1=float(t[-1]));assert np.isfinite(c).all();return abs(c)**2,(tt-t[0])/86400+4,ff
 observed={c:noise[c]+sources[0][c]+sources[1][c] for c in chs};np.savez_compressed(O/'observed.npz',t_lisa=t,**observed)
 maps=compute_self_ae_median_wdm(t,observed['A2'],observed['E2'],resolution=WDMResolution('dt4_M512',4.,512));maps['times_day']=maps['times_day']+4;ev,masks=find_seeded_chirp_events(maps,config=cfg,start_day=4.25,stop_day=5.75,label=mode);ev.to_csv(O/'search.csv',index=False);np.savez_compressed(O/'masks.npz',**masks);assert len(ev)
 union=np.logical_or.reduce(list(masks.values()));used=union.any(axis=1);u=union[used];assert np.all((~u).sum(axis=1)>0);evidence=dict(union_mask=u,used_frequency_indices=np.flatnonzero(used),off_counts=(~u).sum(axis=1));fullnoise={};sums=[]
 for c in chs:
  pn,tt,ff=power(noise[c]);fullnoise[c+'_power']=pn;floor=np.median(pn,axis=1);assert np.all(floor>0);fullnoise[c+'_floor']=floor
  if c in ['A2','E2']:
   for label,s in zip(['mbhb','glitch'],sources):
    ps,_,_=power(s[c]);domain=((ff>=1e-5)&(ff<=.02))[:,None]&((tt>=4.25)&(tt<=5.75))[None,:];strength.append(dict(mode=mode,source=label,channel=c,Q=float((ps/floor[:,None])[domain].sum())));fullnoise[label+'_'+c+'_power']=ps
  if c not in g.SAGNAC_CHANNELS:continue
  q,qt,qf=power(observed[c]);assert np.array_equal(qt,tt) and np.array_equal(tt,maps['times_day']);q=q[used];n=pn[used];nf=floor[used];off=np.array([np.median(v[~mm]) for v,mm in zip(q,u)]);assert np.all(off>0)
  for k,v in [('observed_power',q),('noise_power',n),('noise_floor',nf),('off_floor',off)]:evidence[c+'_'+k]=v
  vals=dict(matched=np.maximum((q-n)/nf[:,None],0),off_subtracted=np.maximum(q/off[:,None]-1,0),noise_off_subtracted=np.maximum(n/off[:,None]-1,0))
  for e in ev.event_id:
   m=masks[e][used];evidence[e+'_mask']=m;sums.append(dict(event_id=e,channel=c,**{k:float(v[m].sum()) for k,v in vals.items()}))
 evidence.update(times_day=tt,frequencies_hz=ff);np.savez_compressed(O/'evidence.npz',**evidence);np.savez_compressed(O/'noise_maps.npz',times_day=tt,frequencies_hz=ff,**fullnoise);pd.DataFrame(sums).to_csv(O/'channel_sums.csv',index=False)
 peak=(sources[0]['t_lisa'][np.argmax(sources[0]['A2']**2+sources[0]['E2']**2)]-t[0])/86400+4
 for rank,r in enumerate(ev.to_dict('records'),1):
  r.update(mode=mode,rank=rank,contains_mbhb_peak=bool(r['start_day']<=peak<=r['stop_day']),overlaps_glitch_input=bool(r['stop_day']>=5.5 and r['start_day']<=5.5+600/86400));sub=[s for s in sums if s['event_id']==r['event_id']]
  for v in vals:
   N=sum(s[v] for s in sub if s['channel'] in g.ZETA_CHANNELS);D=sum(s[v] for s in sub if s['channel'] in g.ABC_CHANNELS);r[v+'_N']=N;r[v+'_D']=D;r[v+'_R']=N/D if D>0 else np.nan
  allrows.append(r)
 band=(ff>=1e-5)&(ff<=.02);v=sum(np.maximum(fullnoise[c+'_power']/fullnoise[c+'_floor'][:,None]-1,0)[band].sum(axis=0) for c in g.ZETA_CHANNELS);noise_rows.extend(dict(mode=mode,time_day=float(x),zeta_noise_excess=float(y)) for x,y in zip(tt,v));print(mode,len(ev),flush=True)
pd.DataFrame(allrows).to_csv(P/'all_candidates.csv',index=False);pd.DataFrame(strength).to_csv(P/'strength.csv',index=False);pd.DataFrame(noise_rows).to_csv(P/'noise_time.csv',index=False);assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items());(P/'analysis_execution.json').write_text(json.dumps(dict(elapsed_s=time.monotonic()-start,maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,input_hashes=hashes,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2));print(pd.DataFrame(allrows)[['mode','rank','start_day','stop_day','off_subtracted_R','matched_R']].to_string(index=False))
