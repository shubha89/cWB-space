"""D014 isolated precessing source study; failure checkpoints preserved."""
import os,sys,json,time,hashlib,argparse,platform,resource,importlib.metadata as md
from pathlib import Path
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
os.environ.setdefault('NUMBA_CACHE_DIR',str(ROOT/'.paper_agent/validation/.numba_cache'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.paper_agent/validation/.mplconfig'))
import numpy as np
from scipy.signal import resample_poly
from phenomxpy import IMRPhenomTPHM
from phenomxpy.utils import MasstoSecond
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
 if out.exists() or not out.is_relative_to(ROOT/'.paper_agent/validation/precessing_earlier_start_v1'):raise ValueError('Require isolated new output')
 out.mkdir(parents=True);started=time.monotonic();manifest=json.loads(args.manifest.read_text());case=manifest['cases'][args.case-1];params=case['params'];report=dict(status='STARTED',case=case,command=sys.argv,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),packages={k:md.version(k) for k in ['numpy','scipy','phenomxpy','wdm-wavelet','lisagwresponse','pytdi','numba']},python=sys.version,platform=platform.platform())
 def checkpoint(stage):report['maxrss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;report['stage']=stage;report['elapsed_s']=time.monotonic()-started;(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(stage,flush=True)
 checkpoint('source_sampling')
 try:
  model=IMRPhenomTPHM(**params)
  old_path=ROOT/'.paper_agent/validation/precessing_v1/case03/polarizations.npz';old=np.load(old_path);old_times=old['times'];ohp,ohc,ot=map(np.asarray,model.compute_polarizations(old_times));assert np.allclose(ot,old_times,atol=1e-8,rtol=0)
  original=np.concatenate([old['hplus'],old['hcross']]);new=np.concatenate([ohp,ohc]);overlap=metric(original,new);overlap['max_abs_difference_over_old_peak']=float(np.max(np.abs(new-original))/np.max(np.hypot(old['hplus'],old['hcross'])));overlap['old_array_sha256']=hashlib.sha256(old_path.read_bytes()).hexdigest();overlap['domain']='all exact saved old model timestamps, no fitting or shifting';report['old_source_comparison']=overlap
  np.savez_compressed(out/'old_source_comparison.npz',times=old_times,new_hplus=ohp,new_hcross=ohc,old_hplus=old['hplus'],old_hcross=old['hcross']);checkpoint('old_source_compared')
  if overlap['relative_l2']>1e-4:raise ValueError('Changed f_min alters common-source polarizations beyond numerical scale')
  t=np.arange(-case['peak_day']*DAY-2400.,(14.-case['peak_day'])*DAY+2400.+2.,2.);tf=np.arange(t[0],t[-1]+1.,1.)
  if len(tf)>1_500_000:raise ValueError('Source support exceeds approved bounded implementation')
  hc0=model.compute_polarizations(t);hp,hc,returned=map(np.asarray,hc0)
  if not np.allclose(returned,t,rtol=0,atol=1e-8):raise ValueError('Returned source times differ')
  fine=IMRPhenomTPHM(**params);hpf,hcf,tr=map(np.asarray,fine.compute_polarizations(tf))
  if not np.allclose(tr,tf,rtol=0,atol=1e-8):raise ValueError('Fine source times differ')
  comparison=metric(np.concatenate([hpf[::2],hcf[::2]]),np.concatenate([hp,hc]));report['sampling_refinement']=comparison
  if comparison['relative_l2']>1e-4:raise ValueError('Cadence refinement exceeds1e-4 raw relative L2')
  if not np.isfinite(hpf).all() or not np.isfinite(hcf).all():raise ValueError('Invalid waveform')
  peak=float(tf[np.argmax(hpf*hpf+hcf*hcf)]);shift=case['peak_day']*DAY-peak
  if tf[0]+shift>-1800 or tf[-1]+shift<14*DAY+1200:raise ValueError('Insufficient finite-observation source margins')
  ft=np.fft.rfftfreq(len(tf),1.);window=np.hanning(len(tf));spec=abs(np.fft.rfft(hpf*window))**2+abs(np.fft.rfft(hcf*window))**2;alias=float(spec[ft>=.125].sum()/spec.sum());report['hann_diagnostic_energy_above_response_nyquist_fraction']=alias
  if alias>1e-6:raise ValueError('Unfiltered source alias risk above1e-6 energy fraction')
  af,bf,gf=fine.compute_euler_angles(tf);report['precession']=dict(chip=float(fine.pPrec.chip),alpha_range_rad=float(np.ptp(af)),beta_range_rad=float(np.ptp(np.arccos(np.clip(bf,-1,1)))));report['source_support']=dict(start_s=float(tf[0]),stop_s=float(tf[-1]),peak_s=peak,shift_s=shift,start_day=float((tf[0]+shift)/DAY),stop_day=float((tf[-1]+shift)/DAY),last_to_peak_amplitude_ratio=float(np.hypot(hpf[-1],hcf[-1])/np.max(np.hypot(hpf,hcf))))
  np.savez_compressed(out/'polarizations.npz',times=tf,hplus=hpf,hcross=hcf,alpha=af,cosbeta=bf,gamma=gf)
  # scipy default Kaiser5 polyphase antialias filter; no fitted amplitude/time shift.
  hpd=resample_poly(hpf,1,4);hcd=resample_poly(hcf,1,4);td=tf[0]+np.arange(len(hpd))*4.
  coarse=np.concatenate([resample_poly(hp,1,2),resample_poly(hc,1,2)]);refined=np.concatenate([hpd,hcd]);report['resampled_refinement_relative_l2']=float(np.linalg.norm(coarse-refined)/np.linalg.norm(refined))
  report['effective_parameters']=dict(inclination=float(model.pWF.inclination),f_ref=float(model.pWF.f_ref),phi_ref=float(model.pWF.phi_ref),s1=model.pWF.s1.tolist(),s2=model.pWF.s2.tolist(),prec_version=model.pPrec.prec_version,final_spin_version=int(model.pPrec.final_spin_version),modes=model.phenTHM.mode_array)
  for key in ['inclination','f_ref','phi_ref','s1','s2']:assert np.array_equal(report['effective_parameters'][key],params[key]),key
  modes,_=fine.compute_CPmodes(tf);freq=np.fft.fftfreq(len(tf),1.);mode_rows=[]
  for key,v in modes.items():
   power=abs(np.fft.fft(v*window))**2;mode_rows.append(dict(mode=key,hann_energy_above_nyquist_fraction=float(power[abs(freq)>=.125].sum()/power.sum())))
  report['mode_spectral_support']=mode_rows
  del modes
  if max(v['hann_energy_above_nyquist_fraction'] for v in mode_rows)>1e-6:raise ValueError('Retained mode aliasing risk above1e-6')
  checkpoint('sampling_verified')
  response.TMP_DIR=out/'tmp';response.INPUT_DIR=out/'inputs';response.OUTPUT_DIR=out;response.PLOT_DIR=out/'plots';response.configure_base();grid=response.reconstruct_checkpoint_grid();noise=response.load_checkpoint_noise(grid)
  response_rel=np.arange(-1800.,14*DAY+1200.+4.,4.);response_times=grid.t_lisa_science[0]+response_rel;target=response_rel-shift
  if target.min()<td[0]+128 or target.max()>td[-1]-128:raise ValueError('Missing antialias margin')
  hpp=np.interp(target,td,hpd);hcp=np.interp(target,td,hcd)
  direct=np.concatenate([np.interp(target,tf,v) for v in [hpf,hcf]]);actual=np.concatenate([hpp,hcp]);report['injected_vs_fine_interpolation_relative_l2']=float(np.linalg.norm(actual-direct)/np.linalg.norm(direct));report['finite_observation']=dict(response_source_start_rel_s=float(response_rel[0]),response_source_stop_rel_s=float(response_rel[-1]),write_start_rel_s=-600.,write_stop_rel_s=float(grid.t_waveform_full[-1]),margin_before_write_s=1200.,margin_after_science_s=1200.,source_nonzero_at_science_start=bool(np.hypot(hpp[np.argmin(abs(response_rel))],hcp[np.argmin(abs(response_rel))])>0),hann_window='spectral diagnostic only; never applied to injected waveform',physical_tmin_s=float(MasstoSecond(model.pWF.tmin,params['total_mass'])))
  np.savez_compressed(out/'injected_strain.npz',times=response_times,hplus=hpp,hcross=hcp)
  if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>4*1024**3:raise ValueError('4GB memory ceiling exceeded')
  checkpoint('response')
  gwfile=out/'source_links.h5';source=ReadStrain(response_times,hpp,hcp,orbits=str(grid.orbits_h5),ra=case['ra'],dec=case['dec']);source.write(str(gwfile),t0=float(grid.t_lisa_full[0]),dt=4.,size=grid.n_total)
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
  old_report=json.loads((ROOT/'.paper_agent/validation/precessing_v1/case03/report.json').read_text());old_event=old_report['comparisons'][0];fixed=((times-t0)/DAY>=old_event['start_day'])&((times-t0)/DAY<=old_event['stop_day']);fixed_x=np.concatenate([product.aet[c][fixed] for c in ['A2','E2']]);fixed_y=np.concatenate([payload['rank1_'+c+'_reconstruction'][fixed] for c in ['A2','E2']]);report['new_rank1_fixed_old_center_domain']=dict(start_day=old_event['start_day'],stop_day=old_event['stop_day'],**metric(fixed_x,fixed_y))
  report['comparisons']=comparisons;report['n_candidates']=len(events);report['status']='COMPLETED_TARGETED_EXAMPLE';report['units']='Hz TDI2, shared checkpoint instrument noise';report['selection']='all observed K32 events, no fitted amplitudes or truth-selected masks';np.savez_compressed(out/'reconstruction.npz',**payload);checkpoint('complete')
 except Exception as error:
  report['status']='BLOCKED';report['error']=repr(error);checkpoint('blocked');raise
if __name__=='__main__':main()
