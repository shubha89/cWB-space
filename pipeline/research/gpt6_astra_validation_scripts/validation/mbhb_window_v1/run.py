"""Approved D028 extension: fixed legacy K256 masks, observed-only growth."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import sys,json,time,csv,hashlib,signal,resource,platform,importlib.metadata as md
from pathlib import Path
from dataclasses import replace
from scipy.ndimage import binary_dilation
import numpy as np
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
os.environ.setdefault('NUMBA_CACHE_DIR',str(OUT/'numba_cache'))
from wdm_wavelet.wdm import WDM
BASE=ROOT/'gpt6_astra_validation_scripts/validation/reconcile_v1/reconstruction_observed';PRODUCT=ROOT/'diagnostics/simulated_2week_mbhb_then_glitch_highrate/outputs/two_week_mbhb_then_glitch_tdi.npz'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check():
 if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>4*1024**3:raise MemoryError('4GB ceiling')
def metric(x,y):
 x=np.concatenate(x);y=np.concatenate(y);a=np.linalg.norm(x);b=np.linalg.norm(y)
 return dict(overlap=float(x@y/a/b),norm_ratio=float(b/a),relative_l2=float(np.linalg.norm(x-y)/a))
def selection(power,t,f,old,row):
 band=(f>=1e-5)&(f<=.1);domain=(t>=float(row['search_start_day']))&(t<=float(row['search_stop_day']))
 threshold=float(np.quantile(power[band][:,domain],.95));assert np.isclose(threshold,float(row['grow_threshold']),rtol=1e-12,atol=0)
 i=np.where(old.any(axis=0))[0];lo=max(0,int(i[0])-2);hi=min(len(t)-1,int(i[-1])+2)
 allowed=np.zeros_like(old);allowed[band,lo:hi+1]=True;eligible=allowed&(power>=threshold)
 footprint=np.zeros((9,3),bool);footprint[:,0]=True;footprint[:,2]=True;footprint[3:6,1]=True
 new=old.copy()
 while True:
  nxt=old|(binary_dilation(new,structure=footprint)&eligible)
  if np.array_equal(nxt,new):break
  new=nxt
 return new,dict(threshold=threshold,threshold_domain_days=[float(row['search_start_day']),float(row['search_stop_day'])],band_hz=[1e-5,.1],old_center_interval_days=[float(t[i[0]]),float(t[i[-1]])],reconstruction_window_days=[float(t[lo]),float(t[hi])],original_pixel_count=int(old.sum()),expanded_pixel_count=int(new.sum()),added_pixel_count=int((new&~old).sum()),disconnected_eligible_count=int((eligible&~new).sum()),added_normalized_power=float(power[new&~old].sum()),disconnected_normalized_power=float(power[eligible&~new].sum()),window_boundary_contact={'left':bool(new[:,lo].any()),'right':bool(new[:,hi].any()),'top_frequency':bool(new[np.where(band)[0][-1]].any())})
def main():
 start=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('30min ceiling')));signal.alarm(1800)
 inputs=[PRODUCT,BASE/'observed_cluster_masks_and_maps.npz',BASE/'observed_cluster_reconstructed_ae_timeseries.npz',BASE/'selected_observed_cluster_events.csv']
 manifest={'command':[sys.executable,*sys.argv],'input_hashes':{str(p.relative_to(ROOT)):sha(p) for p in inputs},'script_sha256':sha(Path(__file__)),'python':sys.version,'platform':platform.platform(),'packages':{x:md.version(x) for x in ['numpy','scipy','wdm-wavelet','numba']},'config':{'M':512,'K':256,'beta_order':6,'precision':6,'backend':'numba','growth_quantile':.95,'time_margin_bins':2,'frequency_band_hz':[1e-5,.1],'normalization':'per-channel full valid-record median','threshold_domain':'original per-case saved known-time search domain','selection_caveat':'Original masks already K256 and selected using catalog-window eligibility; not blind K32 rank1 transfers','units':'Hz','source_generation':False,'seeds':'no new random sampling'}}
 if (OUT/'manifest.json').exists():raise RuntimeError('Fresh outputs required')
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2));report={'status':'RUNNING','cases':[]}
 try:
  products=np.load(PRODUCT);saved=np.load(inputs[1]);series=np.load(inputs[2]);times=products['t_lisa'];rel=products['time_rel_s']/86400;w=WDM(M=512,K=256,beta_order=6,precision=6,backend='numba');maps={};power=0;floors={}
  for c in ['A2','E2']:
   tf=w.t2w(products['observed_'+c],sample_rate=1/np.median(np.diff(times)),t0=float(times[0]));maps[c]=tf;t=(np.arange(tf.n_time)+.5)*tf.dt/86400;f=np.arange(tf.n_freq)*tf.df
   assert np.array_equal(t,saved['times_day']) and np.array_equal(f,saved['freqs_hz']);assert np.all(t*86400<=times[-1]-times[0])
   floor=np.median(np.abs(tf.data)**2,axis=1);assert np.all(np.isfinite(floor)&(floor>0));assert np.allclose(floor,saved['observed_'+c[0]+'_floor'],rtol=1e-12,atol=0)
   floors[c]=floor;power+=np.abs(tf.data/np.sqrt(floor[:,None]))**2
  check()
  for row in csv.DictReader(inputs[3].open()):
   label=row['source_label'];dest=OUT/label;dest.mkdir();old=saved[label+'_cluster'];new,info=selection(power,t,f,old,row)
   np.savez_compressed(dest/'selection.npz',original_mask=old,expanded_mask=new,normalized_power=power,grid_times_day=t,grid_freqs_hz=f,**{c+'_floor':v for c,v in floors.items()})
   info.update(source_label=label,original_event=row,margin_seconds=2*tf.dt,selection_sha256=sha(dest/'selection.npz'))
   (dest/'selection.json').write_text(json.dumps(info,indent=2))
   rec={};oldrec={};controls={}
   for c,tf in maps.items():
    def inv(mask):
     z=replace(tf,data=np.where(mask,tf.data,0j));return .5*(np.asarray(w.w2t(z).value)[:len(times)]+np.asarray(w.w2tQ(z).value)[:len(times)])
    baseline=inv(old);oldrec[c]=series[label+'_'+c+'_cluster_observed_cluster'];delta=float(np.max(np.abs(baseline-oldrec[c])));assert np.allclose(baseline,oldrec[c],rtol=1e-10,atol=1e-18);rec[c]=inv(new);controls[c]={'old_inverse_max_abs_difference':delta}
   truth={c:series[label+'_'+c+'_source'] for c in maps};comparisons={}
   for name,edges in [('original_interval',info['old_center_interval_days']),('expanded_window',info['reconstruction_window_days']),('full_record',[float(rel[0]),float(rel[-1])])]:
    sel=(rel>=edges[0])&(rel<=edges[1]);comparisons[name]={'interval_days':edges,'n_samples':int(sel.sum()),'old':metric([truth[c][sel] for c in maps],[oldrec[c][sel] for c in maps]),'new':metric([truth[c][sel] for c in maps],[rec[c][sel] for c in maps])}
   np.savez_compressed(dest/'waveforms.npz',times=times,time_rel_days=rel,**{c+'_source':v for c,v in truth.items()},**{c+'_old':v for c,v in oldrec.items()},**{c+'_new':v for c,v in rec.items()})
   info.update(comparisons=comparisons,controls=controls,waveforms_sha256=sha(dest/'waveforms.npz'));(dest/'report.json').write_text(json.dumps(info,indent=2));report['cases'].append(info);print(json.dumps({'source':label,'pixels':[info['original_pixel_count'],info['expanded_pixel_count']],'expanded_window':comparisons['expanded_window']}),flush=True);check()
  assert manifest['input_hashes']=={str(p.relative_to(ROOT)):sha(p) for p in inputs};report['status']='COMPLETED'
 except Exception as e:report.update(status='BLOCKED',error=repr(e));raise
 finally:
  report.update(elapsed_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);(OUT/'report.json').write_text(json.dumps(report,indent=2));signal.alarm(0)
if __name__=='__main__':main()
