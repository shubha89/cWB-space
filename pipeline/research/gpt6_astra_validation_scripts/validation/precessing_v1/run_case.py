"""D014 isolated precessing source study; failure checkpoints preserved."""
import os,sys,json,time,hashlib,argparse,platform,importlib.metadata as md
from pathlib import Path
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
os.environ.setdefault('NUMBA_CACHE_DIR',str(ROOT/'gpt6_astra_validation_scripts/validation/.numba_cache'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'gpt6_astra_validation_scripts/validation/.mplconfig'))
import numpy as np
from scipy.signal import resample_poly
from phenomxpy import IMRPhenomTPHM
from lisagwresponse import ReadStrain
from pytdi import Data
from wdm_wavelet.wdm import WDM
import diagnostics.run_generic_burst_response_validation as response
from scripts.blind_event_clustering import compute_self_ae_median_wdm,find_seeded_chirp_events,SeededChirpEventConfig
from scripts.clean_wdm import WDMResolution
DAY=86400.
def metric(x,y):
 a=np.linalg.norm(x);b=np.linalg.norm(y)
 if not np.isfinite(x).all() or not np.isfinite(y).all() or a==0 or b==0:raise ValueError('Nonfinite/zero metric')
 return dict(relative_l2=float(np.linalg.norm(y-x)/a),norm_ratio=float(b/a),normalized_dot=float(np.dot(x,y)/a/b),n_samples=len(x))
def inverse(w,tf):return .5*(w.w2t(tf).value+w.w2tQ(tf).value)
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--case',type=int,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output.resolve()
 if out.exists() or not out.is_relative_to(ROOT/'gpt6_astra_validation_scripts/validation/precessing_v1'):raise ValueError('Require isolated new output')
 out.mkdir(parents=True);started=time.monotonic();manifest=json.loads(args.manifest.read_text());case=manifest['cases'][args.case-1];params=case['params'];report=dict(status='STARTED',case=case,command=sys.argv,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),packages={k:md.version(k) for k in ['numpy','scipy','phenomxpy','wdm-wavelet','lisagwresponse','pytdi','numba']},python=sys.version,platform=platform.platform())
 def checkpoint(stage):report['stage']=stage;report['elapsed_s']=time.monotonic()-started;(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(stage,flush=True)
 checkpoint('source_sampling')
 try:
  model=IMRPhenomTPHM(**params);t=np.arange(-model.pWF.len_neg*2.,model.pWF.len_pos*2.,2.);tf=np.arange(t[0],t[-1]+1.,1.)
  if len(tf)>1_500_000:raise ValueError('Source support exceeds approved bounded implementation')
  hc0=model.compute_polarizations(t);hp,hc,returned=map(np.asarray,hc0)
  if not np.allclose(returned,t,rtol=0,atol=1e-8):raise ValueError('Returned source times differ')
  fine=IMRPhenomTPHM(**params);hpf,hcf,tr=map(np.asarray,fine.compute_polarizations(tf))
  if not np.allclose(tr,tf,rtol=0,atol=1e-8):raise ValueError('Fine source times differ')
  comparison=metric(np.concatenate([hpf[::2],hcf[::2]]),np.concatenate([hp,hc]));report['sampling_refinement']=comparison
  if comparison['relative_l2']>1e-4:raise ValueError('Cadence refinement exceeds1e-4 raw relative L2')
  if not np.isfinite(hpf).all() or not np.isfinite(hcf).all():raise ValueError('Invalid waveform')
  peak=float(tf[np.argmax(hpf*hpf+hcf*hcf)]);shift=case['peak_day']*DAY-peak
  if tf[0]+shift<600 or tf[-1]+shift>14*DAY-600:raise ValueError('Insufficient full source boundary support')
  ft=np.fft.rfftfreq(len(tf),1.);spec=abs(np.fft.rfft(hpf))**2+abs(np.fft.rfft(hcf))**2;alias=float(spec[ft>=.125].sum()/spec.sum());report['unfiltered_energy_above_response_nyquist_fraction']=alias
  if alias>1e-6:raise ValueError('Unfiltered source alias risk above1e-6 energy fraction')
  af,bf,gf=fine.compute_euler_angles(tf);report['precession']=dict(chip=float(fine.pPrec.chip),alpha_range_rad=float(np.ptp(af)),beta_range_rad=float(np.ptp(np.arccos(np.clip(bf,-1,1)))));report['source_support']=dict(start_s=float(tf[0]),stop_s=float(tf[-1]),peak_s=peak,shift_s=shift,start_day=float((tf[0]+shift)/DAY),stop_day=float((tf[-1]+shift)/DAY),last_to_peak_amplitude_ratio=float(np.hypot(hpf[-1],hcf[-1])/np.max(np.hypot(hpf,hcf))))
  np.savez_compressed(out/'polarizations.npz',times=tf,hplus=hpf,hcross=hcf,alpha=af,cosbeta=bf,gamma=gf)
  # scipy default Kaiser5 polyphase antialias filter; no fitted amplitude/time shift.
  hpd=resample_poly(hpf,1,4);hcd=resample_poly(hcf,1,4);td=tf[0]+np.arange(len(hpd))*4.
  response.TMP_DIR=out/'tmp';response.INPUT_DIR=out/'inputs';response.OUTPUT_DIR=out;response.PLOT_DIR=out/'plots';response.configure_base();grid=response.reconstruct_checkpoint_grid();noise=response.load_checkpoint_noise(grid)
  target=grid.t_waveform_full-shift;hpp=np.interp(target,td,hpd,left=0,right=0);hcp=np.interp(target,td,hcd,left=0,right=0)
  checkpoint('response')
  gwfile=out/'source_links.h5';source=ReadStrain(grid.t_lisa_full,hpp,hcp,orbits=str(grid.orbits_h5),ra=case['ra'],dec=case['dec']);source.write(str(gwfile),t0=float(grid.t_lisa_full[0]),dt=4.,size=grid.n_total)
  data=Data.from_gws(str(gwfile),str(grid.orbits_h5),skipped=0,gw_dataset='tcb',orbit_dataset='tcb/ltt');product=response.base.product_from_data(data,grid,multiply_by_central_frequency=True)
  observed={c:noise.aet[c]+product.aet[c] for c in ['A2','E2','T2']};times=grid.t_lisa_science;t0=float(times[0]);maps=compute_self_ae_median_wdm(times,observed['A2'],observed['E2'],resolution=WDMResolution('dt4_M512',4.,512));config=SeededChirpEventConfig(low_hz=1e-5,high_hz=.02,seed_quantile=.99,grow_quantile=.95,max_freq_jump_bins=4,same_time_connect_bins=1,min_pixels=8,min_time_bins=2,freq_tolerance_bins=1.,max_events=30)
  events,masks=find_seeded_chirp_events(maps,config=config,start_day=.5,stop_day=13.5,label=case['id']);events.to_csv(out/'all_candidates.csv',index=False);checkpoint('reconstruction')
  # Freeze and save all discovered masks before any source association.
  np.savez_compressed(out/'discovered_masks.npz',**masks)
  payload={'times':times,'wdm_times_day':maps['times_day'],'wdm_freqs_hz':maps['freqs_hz']};controls=[];w=WDM(M=512,K=256,beta_order=6,precision=6,backend='numba');rawtfs={}
  for c in ['A2','E2']:
   raw=observed[c];wt=w.t2w(raw,sample_rate=.25,t0=t0);wtimes=wt.t0+(np.arange(wt.n_time)+.5)*wt.dt;valid=wtimes<=times[-1]
   if not np.array_equal((wtimes[valid]-t0)/DAY,maps['times_day']) or not np.array_equal(np.arange(wt.n_freq)*wt.df,maps['freqs_hz']):raise ValueError('K grids differ')
   full=inverse(w,wt);rawtfs[c]=wt;payload[c+'_observed']=raw;payload[c+'_full_inverse']=full;controls.append(dict(input=c+'_observed',**metric(raw,full)))
   src=product.aet[c];sf=inverse(w,w.t2w(src,sample_rate=.25,t0=t0));payload[c+'_source']=src;payload[c+'_source_full_inverse']=sf;controls.append(dict(input=c+'_source',**metric(src,sf)))
  report['roundtrips']=controls
  if max(r['relative_l2'] for r in controls)>.1:raise ValueError('Full inverse control failed')
  comparisons=[]
  # All candidates retained; comparison never changes selection or ranks.
  for rank,row in enumerate(events.itertuples(),1):
   local=((times-t0)/DAY>=row.start_day)&((times-t0)/DAY<=row.stop_day);mask=np.zeros(rawtfs['A2'].data.shape,bool);mask[:,valid]=masks[row.event_id];ys=[];xs=[]
   for c in ['A2','E2']:
    rec=inverse(w,replace(rawtfs[c],data=np.where(mask,rawtfs[c].data,0j)));payload[f'rank{rank}_{c}_reconstruction']=rec;ys.append(rec[local]);xs.append(product.aet[c][local])
   x=np.concatenate(xs);y=np.concatenate(ys)
   if np.linalg.norm(x)>0 and np.linalg.norm(y)>0:comparisons.append(dict(rank=rank,event_id=row.event_id,start_day=row.start_day,stop_day=row.stop_day,domain='saved_candidate_center_extent',**metric(x,y)))
  report['comparisons']=comparisons;report['n_candidates']=len(events);report['status']='COMPLETED_TARGETED_EXAMPLE';report['units']='Hz TDI2, shared checkpoint instrument noise';report['selection']='all observed K32 events, no fitted amplitudes or truth-selected masks';np.savez_compressed(out/'reconstruction.npz',**payload);checkpoint('complete')
 except Exception as error:
  report['status']='BLOCKED';report['error']=repr(error);checkpoint('blocked');raise
if __name__=='__main__':main()
