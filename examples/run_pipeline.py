#!/usr/bin/env python3
"""Observed TDI -> K32 candidate search -> optional R_zeta and K256 inverse.

Defaults reproduce the packaged response-study record. Time is seconds; channel
values are TDI frequency fluctuations in Hz. No source waveform is accepted or
used in selection. --config accepts SeededChirpEventConfig fields as JSON.
The inverse is a selected follow-up example, not a reproduction of paper Fig. 7.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, replace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, default=ROOT/'data/pipeline/observed_tdi.npz')
    ap.add_argument('--output-dir', type=Path, required=True, help='Must not exist')
    ap.add_argument('--time-key', default='t_lisa')
    ap.add_argument('--a-key', default='A2')
    ap.add_argument('--e-key', default='E2')
    ap.add_argument('--monitor-keys', nargs=6, default=['alpha2','beta2','gamma2','zeta21','zeta22','zeta23'], metavar='KEY', help='Order: alpha beta gamma zeta1 zeta2 zeta3')
    ap.add_argument('--label', default='mbhb_week1_m1e6_q03_day5p5__native')
    ap.add_argument('--day-offset', type=float, default=4.)
    ap.add_argument('--start-day', type=float, default=4.25)
    ap.add_argument('--stop-day', type=float, default=5.75)
    ap.add_argument('--low-hz', type=float, default=None)
    ap.add_argument('--high-hz', type=float, default=None)
    ap.add_argument('--config', type=Path, help='JSON search dataclass fields; CLI band overrides take precedence')
    ap.add_argument('--response', action='store_true')
    ap.add_argument('--inverse', action='store_true', help='K256 rank-selected connected-window follow-up')
    ap.add_argument('--event-rank', type=int, default=1)
    ap.add_argument('--inverse-guard-start', type=float, default=None)
    ap.add_argument('--inverse-guard-stop', type=float, default=None)
    args = ap.parse_args()
    out = args.output_dir.resolve()
    if out.exists(): ap.error('Output directory already exists; choose a new directory')
    out.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('NUMBA_CACHE_DIR', str(out/'numba_cache'))
    os.environ.setdefault('MPLCONFIGDIR', str(out/'mplconfig'))
    from cwb_space.blind_event_clustering import compute_self_ae_median_wdm, find_seeded_chirp_events, SeededChirpEventConfig
    from cwb_space.clean_wdm import WDMResolution, compute_wdm_transform, require_wdm_dependencies
    from cwb_space.response import ABC_CHANNELS, ZETA_CHANNELS, observed_background_ratio
    from cwb_space.reconstruction import select_reconstruction_window, inverse_quadrature_average
    import numpy as np
    started = time.monotonic()
    report = {'status':'RUNNING','input_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),
              'input_name':args.input.name,'time_key':args.time_key,'A_key':args.a_key,'E_key':args.e_key,
              'monitor_keys':args.monitor_keys,'day_offset':args.day_offset,'search_days':[args.start_day,args.stop_day],
              'scope':'Observed TDI only. No template input. Optional inverse is a new follow-up example, not Fig7 reproduction.'}
    try:
        with np.load(args.input, allow_pickle=False) as z:
            t=np.asarray(z[args.time_key], dtype=float)
            obs={c:np.asarray(z[k], dtype=float) for c,k in [('A2',args.a_key),('E2',args.e_key)]}
            if args.response: obs.update({c:np.asarray(z[k],dtype=float) for c,k in zip((*ABC_CHANNELS,*ZETA_CHANNELS),args.monitor_keys)})
        if t.ndim!=1 or len(t)<2 or not np.isfinite(t).all(): raise ValueError('Finite 1D time grid required')
        dt=float(np.median(np.diff(t)))
        if dt<=0 or not np.allclose(np.diff(t),dt,rtol=1e-9,atol=1e-8): raise ValueError('Strictly increasing uniformly sampled time required')
        for c,x in obs.items():
            if x.shape!=t.shape or not np.isfinite(x).all(): raise ValueError('Finite channels matching time required: '+c)
        config=asdict(SeededChirpEventConfig(high_hz=.02,max_events=30))
        if args.config: config.update(json.loads(args.config.read_text()))
        if args.low_hz is not None: config['low_hz']=args.low_hz
        if args.high_hz is not None: config['high_hz']=args.high_hz
        cfg=SeededChirpEventConfig(**config)
        if not 0<=cfg.low_hz<cfg.high_hz<=.5/dt: raise ValueError('Band must lie within sampled Nyquist range')
        if args.start_day>=args.stop_day: raise ValueError('Invalid search guard')
        maps=compute_self_ae_median_wdm(t,obs['A2'],obs['E2'],resolution=WDMResolution('observed_M512',dt,512))
        maps['times_day']+=args.day_offset
        ev,masks=find_seeded_chirp_events(maps,config=cfg,start_day=args.start_day,stop_day=args.stop_day,label=args.label)
        np.savez_compressed(out/'search_maps.npz',**maps)
        np.savez_compressed(out/'masks.npz',**masks)
        ev.to_csv(out/'candidates.csv',index=False)
        report.update(configuration=config,sample_dt_s=dt,time_start_s=float(t[0]),time_stop_s=float(t[-1]),n_samples=len(t),candidates=len(ev),search_K=32,M=512,beta_order=6,precision=6)
        _,WDM=require_wdm_dependencies()
        if args.response and masks:
            w=WDM(M=512,K=32,beta_order=6,precision=6,backend='numba')
            used=np.logical_or.reduce(list(masks.values())).any(axis=1);powers={}
            for c in (*ABC_CHANNELS,*ZETA_CHANNELS):
                co,at,af,_,_=compute_wdm_transform(w,obs[c],sample_rate=1/dt,t0=float(t[0]),original_t1=float(t[-1]))
                np.testing.assert_array_equal((at-t[0])/86400+args.day_offset,maps['times_day'])
                np.testing.assert_array_equal(af,maps['freqs_hz'])
                powers[c]=(abs(co)**2)[used]
            ratios,sums,floors=observed_background_ratio(powers,{k:v[used] for k,v in masks.items()})
            ratios.to_csv(out/'response.csv',index=False);sums.to_csv(out/'channel_sums.csv',index=False)
            np.savez_compressed(out/'monitor_powers.npz',used_frequency_indices=np.flatnonzero(used),**{c+'_power':v for c,v in powers.items()},**{c+'_floor':v for c,v in floors.items()})
        if args.inverse:
            if not 1<=args.event_rank<=len(ev): raise ValueError('Requested inverse rank absent')
            event_id=ev.iloc[args.event_rank-1].event_id;original=masks[event_id]
            w=WDM(M=512,K=256,beta_order=6,precision=6,backend='numba');transforms={};power=0;floors={};pads={}
            for c in ['A2','E2']:
                co,at,af,tf,pad=compute_wdm_transform(w,obs[c],sample_rate=1/dt,t0=float(t[0]),original_t1=float(t[-1]))
                np.testing.assert_array_equal((at-t[0])/86400+args.day_offset,maps['times_day']);np.testing.assert_array_equal(af,maps['freqs_hz'])
                raw=abs(co)**2;floor=np.median(raw,axis=1)
                if not np.all(np.isfinite(floor)&(floor>0)): raise ValueError('Nonpositive K256 floor; no replacement')
                power+=abs(co/np.sqrt(floor[:,None]))**2;floors[c]=floor;transforms[c]=tf;pads[c]=pad
            gs=args.start_day if args.inverse_guard_start is None else args.inverse_guard_start
            ge=args.stop_day if args.inverse_guard_stop is None else args.inverse_guard_stop
            band=(maps['freqs_hz']>=cfg.low_hz)&(maps['freqs_hz']<=cfg.high_hz)
            guard=(maps['times_day']>=gs)&(maps['times_day']<=ge)
            if not np.any(guard): raise ValueError('Empty inverse quantile guard')
            expanded,threshold,window=select_reconstruction_window(power,original,band,guard)
            np.savez_compressed(out/'inverse_selection.npz',original_mask=original,expanded_mask=expanded,window=window,normalized_power=power,times_day=maps['times_day'],freqs_hz=maps['freqs_hz'],**{c+'_floor':v for c,v in floors.items()})
            reconstructed={}
            for c,tf in transforms.items():
                centers=tf.t0+(np.arange(tf.n_time)+.5)*tf.dt;valid=centers<=t[-1]
                full=np.zeros(tf.data.shape,bool);full[:,valid]=expanded
                x=inverse_quadrature_average(w,replace(tf,data=np.where(full,tf.data,0j)))
                if len(x)<len(t) or not np.isfinite(x).all(): raise ValueError('Invalid inverse output')
                reconstructed[c+'_reconstructed']=x[:len(t)]
            np.savez_compressed(out/'reconstruction.npz',t_lisa=t,**reconstructed)
            report['inverse']={'event_id':event_id,'K':256,'margin_bins':2,'growth_quantile':.95,'threshold':threshold,'guard_days':[gs,ge],'right_zero_padding_samples':pads,'original_pixels':int(original.sum()),'expanded_pixels':int(expanded.sum()),'normalization':'Observed per-channel full-valid-record medians; raw coefficients inverted; quadrature average; cropped to input length','limitation':'Short-record example; any additional padding is recorded; finite inverse residuals remain; no source truth or paper-reconstruction equivalence asserted.'}
        report['status']='COMPLETED'
    except Exception as exc:
        report.update(status='FAILED',error=repr(exc));raise
    finally:
        report['elapsed_s']=time.monotonic()-started
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__': main()
