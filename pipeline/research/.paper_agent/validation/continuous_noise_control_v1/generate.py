from pathlib import Path
import os,sys,time,json,resource,warnings,hashlib
P=Path(__file__).resolve().parent;ROOT=P.parents[2];sys.path.insert(0,str(ROOT));os.environ.setdefault('MPLCONFIGDIR',str(P.parent/'.mplconfig'))
import numpy as np
from lisainstrument import Instrument
from pytdi import Data
from scipy.signal import resample_poly
from scripts.six_link_generation import LisaLinkSimulationConfig
from scripts.tdi_computation import _evaluate_michelson_tdi,_evaluate_sagnac_tdi,apply_frequency_cut
start=time.monotonic();D=86400.;dt=.1;factor=40;lo=1e-5;hi=.1;seed=752001
orbit=ROOT/'diagnostics/simulated_2week_mbhb_then_glitch_highrate/inputs/orbits_2week.h5';old=ROOT/'diagnostics/simulated_2week_mbhb_then_glitch_highrate/outputs/mbhb_week1_sagnac_zeta_products.npz';week=np.load(old)['t_lisa'];week0=float(week[0]);a=4*D-3600;b=6*D+3000;n=int(round((b-a)/dt));h5=P/'instrument.h5'
assert not h5.exists();print('Generate continuous instrument',n,'samples',flush=True)
instrument=Instrument(size=n,dt=dt,t0=week0+a,orbits=str(orbit),gws=None,orbit_dataset='tcb/ltt',central_freq=2.816e14,seed=seed,noises_f_min_hz=lo,clock_f_min_hz=lo,chunk_gw_files=True)
with warnings.catch_warnings():
 warnings.simplefilter('ignore');instrument.export_hdf5(path=str(h5),overwrite=False)
del instrument
print('Instrument exported',time.monotonic()-start,flush=True)
data=Data.from_instrument(str(h5),signals='fluctuations',skipped=0)
cfg=LisaLinkSimulationConfig(dt=dt,science_duration=b-a,burn_in_seconds=0.,central_freq_hz=2.816e14,orbit_dt=600.,frequency_low_hz=lo,frequency_high_hz=hi,tdi_generation=2,interpolation_order=31,tdi_unit='frequency',orbit_dataset='tcb/ltt',gw_dataset='tcb')
x,y,z,aa,ee,tt,_=_evaluate_michelson_tdi(data,cfg);raw={'A2':aa,'E2':ee,'T2':tt};del x,y,z
print('Michelson complete',time.monotonic()-start,flush=True)
sag,_=_evaluate_sagnac_tdi(data,cfg);raw.update(sag);del data
print('All TDI complete',time.monotonic()-start,flush=True)
# Remove initialization600s; retain3000s padding for continuous filtering.
init=int(round(600/dt));start_sc=int(round((4*D-a)/dt));end_sc=int(round((6*D-a)/dt));crop_low=3000//4;nout=int(2*D/4)
continuous={};split={};checks={}
for c,v in raw.items():
 v=np.asarray(v);assert len(v)==n and np.isfinite(v).all();np.save(P/('raw_'+c+'.npy'),v)
 whole=resample_poly(apply_frequency_cut(v[init:],dt,lo,hi),up=1,down=factor);continuous[c]=whole[crop_low:crop_low+nout]
 pieces=[]
 for day in [4,5]:
  i=int(round((day*D-a)/dt));j=i+int(D/dt);pieces.append(resample_poly(apply_frequency_cut(v[i:j],dt,lo,hi),up=1,down=factor))
 split[c]=np.concatenate(pieces);assert len(continuous[c])==nout and len(split[c])==nout
 checks[c]=dict(continuous_std=float(np.std(continuous[c])),split_std=float(np.std(split[c])))
 assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<16*1024**3
science_t=week0+np.arange(4*D,6*D,4.)
np.savez_compressed(P/'noise_continuous.npz',t_lisa=science_t,**continuous);np.savez_compressed(P/'noise_split.npz',t_lisa=science_t,**split)
(P/'generation.json').write_text(json.dumps(dict(seed=seed,high_dt=dt,analysis_dt=4,science_days=[4,6],raw_start_day=a/D,raw_stop_day=b/D,raw_n=n,week0=week0,initialization_removed_s=600,filter_padding_s=3000,elapsed_s=time.monotonic()-start,maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,checks=checks,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),orbit_sha256=hashlib.sha256(orbit.read_bytes()).hexdigest(),old_grid_sha256=hashlib.sha256(old.read_bytes()).hexdigest()),indent=2));print('COMPLETE',time.monotonic()-start,flush=True)
