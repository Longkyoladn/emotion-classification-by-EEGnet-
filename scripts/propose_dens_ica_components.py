"""Checkpoint P6: fit ICA and propose artifact components without applying removal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.signal import welch


EEG = [f"E{i}" for i in range(1, 129)]
BADS = ["E50", "E103"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--subject", default="sub-mit003")
    p.add_argument("--output-dir", type=Path, default=Path("reports/qc"))
    args = p.parse_args()

    mf = pd.read_csv(args.manifest)
    trials = mf[mf.subject.eq(args.subject) & mf.match_status.eq("matched") & mf.validation_issues.fillna("").eq("")].sort_values("event_trial_index")
    raw = mne.io.read_raw_eeglab(args.dataset_root / trials.eeg_set_path.dropna().unique()[0], preload=False, verbose="ERROR")
    mapping = {**{c:"eeg" for c in EEG}, "E129":"misc", "ECG":"ecg", "EMG":"emg", "EMG_2":"emg"}
    raw.set_channel_types(mapping, verbose="ERROR")
    raw.set_montage(mne.channels.make_standard_montage("GSN-HydroCel-128"), on_missing="ignore", verbose="ERROR")
    sfreq = float(raw.info["sfreq"]); pad = int(15*sfreq)
    segments = []
    aux_segments = []
    for tr in trials.itertuples(index=False):
        start, stop = int(round(tr.onset_seconds*sfreq)), int(round(tr.end_seconds*sfreq))
        ps, pe = max(0,start-pad), min(raw.n_times,stop+pad)
        seg = raw.copy().crop(ps/sfreq,(pe-1)/sfreq,include_tmax=True).load_data(verbose="ERROR")
        seg.notch_filter([50.], phase="zero", verbose="ERROR").filter(1.,45., phase="zero", verbose="ERROR")
        seg.crop((start-ps)/sfreq,(stop-ps-1)/sfreq,include_tmax=True)
        aux_segments.append(seg.get_data(picks=["ECG","EMG","EMG_2"]))
        eeg = seg.copy().pick(EEG); eeg.info["bads"]=BADS
        eeg.interpolate_bads(reset_bads=False, method={"eeg":"spline"}, verbose="ERROR")
        eeg.set_eeg_reference("average", projection=False, verbose="ERROR")
        segments.append(eeg)

    concat = mne.concatenate_raws(segments, preload=True, verbose="ERROR")
    aux = np.concatenate(aux_segments,axis=1)
    # Retain 99% PCA variance; exclude interpolated channels from fitting.
    ica = mne.preprocessing.ICA(n_components=0.99, method="fastica", random_state=42, max_iter=1000)
    ica.fit(concat, picks="eeg", reject_by_annotation=True, verbose="ERROR")
    sources = ica.get_sources(concat).get_data()

    rows=[]
    for i,src in enumerate(sources):
        corrs=[float(np.corrcoef(src,a)[0,1]) for a in aux]
        f,power=welch(src,fs=sfreq,nperseg=int(4*sfreq),noverlap=int(2*sfreq))
        total=np.trapezoid(power[(f>=1)&(f<=45)],f[(f>=1)&(f<=45)])
        low=np.trapezoid(power[(f>=1)&(f<4)],f[(f>=1)&(f<4)])/max(total,1e-20)
        high=np.trapezoid(power[(f>=30)&(f<=45)],f[(f>=30)&(f<=45)])/max(total,1e-20)
        kurt=float(pd.Series(src).kurt())
        reasons=[]
        if abs(corrs[0])>=.3: reasons.append("ecg_correlation")
        if max(abs(corrs[1]),abs(corrs[2]))>=.3: reasons.append("emg_correlation")
        # Kurtosis alone is expected for non-Gaussian ICA sources and must not
        # be used as a rejection rule. Require spectral corroboration.
        if high>=.40 and abs(kurt)>=15:
            reasons.append("high_frequency_power_with_kurtosis")
        rows.append({"component":i,"ecg_correlation":corrs[0],"emg_correlation":corrs[1],"emg2_correlation":corrs[2],"low_frequency_fraction":low,"high_frequency_fraction":high,"kurtosis":kurt,"proposed_artifact":bool(reasons),"reasons":";".join(reasons)})
    table=pd.DataFrame(rows)
    proposed=table.loc[table.proposed_artifact,"component"].astype(int).tolist()
    out=args.output_dir/args.subject; out.mkdir(parents=True,exist_ok=True)
    table.to_csv(out/"p6_ica_component_metrics.csv",index=False)
    ica.save(out/"p6_ica_solution-ica.fif",overwrite=True)

    # Review figures: topographies and spectra for all proposed components.
    if proposed:
        figs=ica.plot_components(picks=proposed,inst=concat,show=False)
        if not isinstance(figs,list): figs=[figs]
        for j,fig in enumerate(figs): fig.savefig(out/f"p6_proposed_topographies_{j+1}.png",dpi=150); plt.close(fig)
        fig,axes=plt.subplots(len(proposed),1,figsize=(10,max(3,2.2*len(proposed))),squeeze=False)
        for ax,idx in zip(axes[:,0],proposed):
            f,pw=welch(sources[idx],fs=sfreq,nperseg=int(4*sfreq)); mask=(f>=1)&(f<=45)
            ax.semilogy(f[mask],pw[mask]); ax.set(title=f"IC {idx}: {table.loc[idx,'reasons']}",xlabel="Hz",ylabel="PSD"); ax.grid(alpha=.25)
        fig.tight_layout(); fig.savefig(out/"p6_proposed_component_spectra.png",dpi=150); plt.close(fig)

    report={"checkpoint":"P6","subject":args.subject,"trials":int(len(trials)),"fit_highpass_hz":1.0,"fit_lowpass_hz":45.0,"notch_hz":50.0,"ica_method":"fastica","random_state":42,"pca_variance_target":0.99,"components":int(ica.n_components_),"review_candidates":proposed,"proposal_thresholds":{"absolute_ecg_correlation":.3,"absolute_emg_correlation":.3,"high_frequency_fraction_with_absolute_kurtosis":{"high_frequency_fraction":.40,"absolute_kurtosis":15},"kurtosis_alone_is_not_a_rule":True},"ica_applied_to_eeg":False,"raw_file_unchanged":True,"operations_not_performed":["component removal","saving cleaned EEG","trial rejection","label derivation"]}
    (out/"p6_ica_summary.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
