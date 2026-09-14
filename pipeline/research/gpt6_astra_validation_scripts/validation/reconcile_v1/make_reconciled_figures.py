"""Render existing selections and reproduced raw reconstruction; no fitting/tuning."""
from pathlib import Path
import importlib.util,json,hashlib,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'gpt6_astra_validation_scripts/figures/reconcile_v1'
OUT.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('original_paper_figures',ROOT/'docs/paper/cwb_space_prd/make_paper_figures.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def save(fig,stem):
    names={'fig03_primary_validation_summary':'primary_reconciled','fig05_generic_burst_summary':'generic_reconciled'}
    # Label pixelwise fraction correctly; common physical [0,1] scale exposes all values.
    for ax in fig.axes:
        if 'Zeta fraction' in ax.get_title():
            ax.set_title('Pixel-wise zeta fraction on selected A/E pixels',fontsize=10,pad=2)
            for coll in ax.collections:coll.set_clim(0,1)
        if 'E_\\zeta' in ax.get_ylabel():ax.set_ylabel(r'$F_\zeta(t,f)$',fontsize=9)
    for ax in fig.axes:
        for coll in ax.collections:
            if type(coll).__name__ == 'QuadMesh':coll.set_rasterized(True)
    fig.savefig(OUT/(names[stem]+'.pdf'),bbox_inches='tight',dpi=300)
    fig.savefig(OUT/(names[stem]+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
m.save_figure=save
if '--diagnostics' in sys.argv:
    m.make_primary_validation_summary();m.make_generic_burst_response()
if '--reconstruction' in sys.argv:
    import csv
    run=ROOT/'gpt6_astra_validation_scripts/validation/reconcile_v1/reconstruction_observed'
    data=np.load(run/'observed_cluster_reconstructed_ae_timeseries.npz')
    days=data['time_day'];metrics=[]
    fig,axs=plt.subplots(3,2,figsize=(8.3,7.4),layout='constrained')
    for i in range(3):
        src=f'mbhb{i+1}'
        truth=np.concatenate([data[f'{src}_{c}_source'] for c in ['A2','E2']])
        raw=np.concatenate([data[f'{src}_{c}_cluster_observed_cluster'] for c in ['A2','E2']])
        match=float(np.dot(truth,raw)/np.sqrt(np.dot(truth,truth)*np.dot(raw,raw)))
        err=float(np.linalg.norm(raw-truth)/np.linalg.norm(truth))
        norm=float(np.linalg.norm(raw)/np.linalg.norm(truth))
        metrics.append(dict(source_label=src,match=match,rel_l2=err,norm_over_source=norm))
        peak=int(np.argmax(data[f'{src}_A2_source']**2+data[f'{src}_E2_source']**2))
        show=np.abs(days-days[peak])<=.008
        for j,c in enumerate(['A2','E2']):
            ax=axs[i,j];a=data[f'{src}_{c}_source'];b=data[f'{src}_{c}_cluster_observed_cluster']
            ax.plot(days[show],a[show],color='#202a35',lw=1,label='Injected source')
            ax.plot(days[show],b[show],color='#c54e22',lw=.9,ls='--',label='Observed-cluster reconstruction')
            ax.set_title(f'MBHB {i+1}, {c}',fontsize=10)
            ax.set_ylabel(f'{c} [Hz]');ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useMathText=True)
            ax.ticklabel_format(axis='x',style='plain',useOffset=False);ax.grid(alpha=.15)
            ax.tick_params(labelsize=8)
            if i==2:ax.set_xlabel('Time from data start [days]')
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=2,frameon=False,fontsize=9)
    for ext in ['png','pdf']:fig.savefig(OUT/f'observed_reconstruction.{ext}',dpi=180)
    plt.close(fig)
    (run/'independent_raw_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
