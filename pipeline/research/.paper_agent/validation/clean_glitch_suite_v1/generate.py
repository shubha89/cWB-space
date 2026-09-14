from pathlib import Path
import sys,os,time,json,hashlib,resource
from dataclasses import replace,asdict
P=Path(__file__).resolve().parent;ROOT=P.parents[2];sys.path.insert(0,str(ROOT));os.environ.setdefault('MPLCONFIGDIR',str(P.parent/'.mplconfig'));os.environ.setdefault('NUMBA_CACHE_DIR',str(P.parent/'.numba_cache'))
import numpy as np
import diagnostics.run_multilink_glitch_detection_rzeta_study as l
import diagnostics.run_2week_mbhb_zeta_wdm_analysis as z
import diagnostics.generate_2week_mbhb_then_glitch_highrate_tdi as primary
import diagnostics.run_generic_burst_response_validation as g
b=l.base;start=time.monotonic();S=P/'sources';S.mkdir(exist_ok=True);TMP=P/'tmp';TMP.mkdir(exist_ok=True)
b.TMP_DIR=TMP;g.TMP_DIR=TMP;b.DT=4.;b.BURN_IN_SECONDS=600.;b.ORBIT_DT=600.;b.LOW_HZ=1e-5;b.HIGH_HZ=.1;b.CENTRAL_FREQ_HZ=2.816e14
t=np.load(l.PRODUCTS_NPZ)['t_lisa'];idx=slice(86400,129600);records=[]
def save(cid,group,kind,product,params,extra=None):
 a={'t_lisa':product.t_lisa,**product.aet,**product.sagnac};power=product.aet['A2']**2+product.aet['E2']**2;peak=(product.t_lisa[np.argmax(power)]-t[0])/86400;fraction=float(power[idx].sum()/power.sum());out={c:v[idx] for c,v in a.items()};np.savez_compressed(S/(cid+'.npz'),**out);r=dict(case_id=cid,group=group,kind=kind,parameters=params,source_path=str(S/(cid+'.npz')),ae_peak_day=float(peak),ae_timepower_fraction_two_days=fraction,full_source_samples=len(power),extra=extra or {});records.append(r);(P/'manifest.json').write_text(json.dumps(records,indent=2));print('DONE',cid,'coverage',fraction,'elapsed',time.monotonic()-start,flush=True);assert time.monotonic()-start<2700 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<8*1024**3
for original in primary.MBHB_CASES:
 cid=original.case_id;case=replace(original,peak_day=4.5)
 if case is None:raise ValueError('Inspect MBHB epoch field')
 parts={}
 for days in [14,7]:
  b.SCIENCE_DURATION=days*86400.;grid=z.make_week1_grid(t[:int(days*86400/4)]);parts[days],window=b.generate_mbhb_gw_product(case,grid)
 prod=b.TDIProduct(t_lisa=parts[7].t_lisa,aet={c:v[:len(parts[7].t_lisa)] for c,v in parts[14].aet.items()},sagnac=parts[7].sagnac)
 save(cid,'primary','mbhb',prod,asdict(case),dict(effective_inclination_rad=0.,AE_record_days=14,Sagnac_record_days=7))
for group,cases in [('primary',primary.GLITCH_CASES),('generic',g.GLITCH_CASES)]:
 for original in cases:
  case=replace(original,inj_day=5.5);parts={}
  for days in ([14,7] if group=='primary' else [14]):
   b.SCIENCE_DURATION=days*86400.;grid=z.make_week1_grid(t[:int(days*86400/4)]);parts[days],window=b.generate_glitch_product(case,grid)
  if group=='primary':prod=b.TDIProduct(t_lisa=parts[7].t_lisa,aet={c:v[:len(parts[7].t_lisa)] for c,v in parts[14].aet.items()},sagnac=parts[7].sagnac)
  else:prod=parts[14]
  save(case.case_id,group,'glitch',prod,asdict(case),dict(input_intervals=[[5.5,5.5+case.width_s/86400]]))
for original in g.SIGNAL_CASES:
 case=replace(original,center_day=4.5);b.SCIENCE_DURATION=14*86400.;grid=z.make_week1_grid(t);prod,meta=g.generate_response_burst_product(case,grid);save(case.case_id,'generic',case.morphology,prod,asdict(case),dict(generator_metadata=meta))
b.SCIENCE_DURATION=7*86400.;grid=z.make_week1_grid(t[:151200])
for original in l.GLITCH_CASES:
 ref=float(np.median([c.local_peak_day for c in original.components]));components=[replace(c,local_peak_day=5.5+c.local_peak_day-ref) for c in original.components]
 if original.case_id=='single_12_narrow_high_day8p2':
  cached=l.load_cached_product(P.parent/'half_day_control_v1/sources/glitch.npz');prod=b.TDIProduct(t_lisa=cached.t_lisa,aet={c:v/2.521015903059943 for c,v in cached.aet.items()},sagnac={c:v/2.521015903059943 for c,v in cached.sagnac.items()});reuse='D059 source scaled back to native level'
 else:
  pieces=[]
  for i,c in enumerate(components):
   case=b.GlitchCase(original.case_id+'_'+str(i),c.injection_point,c.local_peak_day,c.width_s,c.level_hz,original.noise_seed);prod,_=b.generate_glitch_product(case,grid);pieces.append(prod)
  prod=l.sum_tdi_products(pieces);reuse='regenerated epoch-dependent response'
 save(original.case_id,'multilink','glitch',prod,dict(components=[asdict(c) for c in components]),dict(input_intervals=[[c.local_peak_day,c.local_peak_day+c.width_s/86400] for c in components],reuse=reuse))
(P/'generation.json').write_text(json.dumps(dict(elapsed_s=time.monotonic()-start,maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),n_sources=len(records)),indent=2))
