"""Approved D014 sampling/mode support checks; no parameter changes."""
import sys,os,json,time,hashlib,resource
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
from phenomxpy import IMRPhenomTPHM
r=Path(__file__).resolve().parent;c=r/'case03';p=json.loads((r/'manifest.json').read_text())['cases'][2]['params'];z=np.load(c/'polarizations.npz');t=z['times'];hp=z['hplus'];hc=z['hcross'];model=IMRPhenomTPHM(**p);began=time.monotonic()
def l2(x,y):return float(np.linalg.norm(x-y)/np.linalg.norm(x))
eff=dict(inclination=float(model.pWF.inclination),f_ref=float(model.pWF.f_ref),phi_ref=float(model.pWF.phi_ref),s1=model.pWF.s1.tolist(),s2=model.pWF.s2.tolist(),prec_version=model.pPrec.prec_version,final_spin_version=int(model.pPrec.final_spin_version),modes=model.phenTHM.mode_array)
for k in ['inclination','f_ref','phi_ref','s1','s2']:assert np.array_equal(eff[k],p[k]),k
assert eff['prec_version']=='nnlo'
# Coarse values exactly reproduced fine[::2] in previous checkpoint; now compare resampled outputs.
a=np.concatenate([resample_poly(hp,1,4),resample_poly(hc,1,4)]);b=np.concatenate([resample_poly(hp[::2],1,2),resample_poly(hc[::2],1,2)]);sampling=l2(a,b)
shift=json.loads((c/'report.json').read_text())['source_support']['shift_s'];target=np.arange(0,14*86400,4.)-shift;td=t[0]+np.arange(len(resample_poly(hp,1,4)))*4
injected=np.concatenate([np.interp(target,td,resample_poly(v,1,4),left=0,right=0) for v in [hp,hc]]);fine=np.concatenate([np.interp(target,t,v,left=0,right=0) for v in [hp,hc]]);interp=l2(fine,injected)
modes,mt=model.compute_CPmodes(t);freq=np.fft.fftfreq(len(t),1.);rows=[]
for key,v in modes.items():
 power=abs(np.fft.fft(v))**2;rows.append(dict(mode=key,energy_above_response_nyquist_fraction=float(power[abs(freq)>=.125].sum()/power.sum())))
root=r.parents[2];paths=[root/'diagnostics/simulated_2week_mbhb_then_glitch_highrate/inputs/orbits_2week.h5',root/'diagnostics/simulated_2week_mbhb_then_glitch_highrate/outputs/two_week_mbhb_then_glitch_tdi.npz'];hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}
result=dict(effective_parameters=eff,resampled_1s_vs_2s_relative_l2=sampling,injected_4s_vs_fine_interpolation_relative_l2=interp,mode_spectral_support=rows,input_hashes=hashes,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seconds=time.monotonic()-began,maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,command=sys.argv)
(c/'sampling_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
