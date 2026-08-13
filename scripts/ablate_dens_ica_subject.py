"""P6b: subject-specific ICA ablation and conservative automatic decision."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import mne, numpy as np, pandas as pd
from scipy.signal import welch

EEG=[f"E{i}" for i in range(1,129)]
BANDS={"delta":(1,4),"theta":(4,8),"alpha":(8,13),"beta":(13,30),"gamma":(30,45)}
def powers(x,sf):
 f,p=welch(x,fs=sf,nperseg=int(4*sf),noverlap=int(2*sf),axis=-1)
 return {n:float(np.median(np.trapezoid(p[:,(f>=lo)&(f<hi)],f[(f>=lo)&(f<hi)],axis=1))) for n,(lo,hi) in BANDS.items()}

def main():
 p=argparse.ArgumentParser(); p.add_argument("--dataset-root",type=Path,required=True); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--ica",type=Path,required=True); p.add_argument("--subject",required=True); p.add_argument("--output-dir",type=Path,default=Path("reports/qc")); a=p.parse_args()
 out=a.output_dir/a.subject; badtab=pd.read_csv(out/"p4_bad_channel_decisions.csv"); comps=pd.read_csv(out/"p6_ica_component_metrics.csv")
 bads=badtab.loc[badtab.decision.eq("bad"),"channel"].tolist(); candidates=comps.loc[comps.proposed_artifact,"component"].astype(int).tolist()
 physiological=comps.loc[comps[["ecg_correlation","emg_correlation","emg2_correlation"]].abs().max(axis=1).ge(.30),"component"].astype(int).tolist()
 strategies={"no_removal":[],"physiological_only":physiological,"all_candidates":candidates}
 mf=pd.read_csv(a.manifest); trials=mf[mf.subject.eq(a.subject)&mf.match_status.eq("matched")&mf.validation_issues.fillna("").eq("")].sort_values("event_trial_index")
 raw=mne.io.read_raw_eeglab(a.dataset_root/trials.eeg_set_path.dropna().unique()[0],preload=False,verbose="ERROR"); raw.set_channel_types({**{c:"eeg" for c in EEG},"E129":"misc","ECG":"ecg","EMG":"emg","EMG_2":"emg"},verbose="ERROR"); raw.set_montage(mne.channels.make_standard_montage("GSN-HydroCel-128"),on_missing="ignore",verbose="ERROR")
 ica=mne.preprocessing.read_ica(a.ica,verbose="ERROR"); sf=float(raw.info["sfreq"]); pad=int(15*sf); rows=[]
 for tr in trials.itertuples(index=False):
  start,stop=int(round(tr.onset_seconds*sf)),int(round(tr.end_seconds*sf)); ps,pe=max(0,start-pad),min(raw.n_times,stop+pad)
  seg=raw.copy().crop(ps/sf,(pe-1)/sf,include_tmax=True).load_data(verbose="ERROR"); seg.notch_filter([50.],phase="zero",verbose="ERROR").filter(.5,45.,phase="zero",verbose="ERROR"); seg.crop((start-ps)/sf,(stop-ps-1)/sf,include_tmax=True)
  aux=seg.get_data(picks=["ECG","EMG","EMG_2"]); base=seg.copy().pick(EEG); base.info["bads"]=bads
  if bads: base.interpolate_bads(reset_bads=False,method={"eeg":"spline"},verbose="ERROR")
  base.set_eeg_reference("average",projection=False,verbose="ERROR"); x0=base.get_data(); p0=powers(x0,sf)
  for name,excluded in strategies.items():
   cleaned=base.copy(); ica.apply(cleaned,exclude=excluded,verbose="ERROR"); x=cleaned.get_data(); corr=max(abs(np.corrcoef(ch,auxch)[0,1]) for ch in x for auxch in aux); pw=powers(x,sf)
   rows.append({"trial":int(tr.event_trial_index),"strategy":name,"components":";".join(map(str,excluded)),"max_aux_correlation":corr,"relative_rms_change":float(np.sqrt(np.mean((x-x0)**2))/max(np.sqrt(np.mean(x0**2)),1e-20)),**{f"{b}_retention":pw[b]/max(p0[b],1e-20) for b in BANDS}})
 detail=pd.DataFrame(rows); summaries=[]
 for name,g in detail.groupby("strategy",sort=False): summaries.append({"strategy":name,"components":g.components.iloc[0],"median_max_aux_correlation":float(g.max_aux_correlation.median()),"median_relative_rms_change":float(g.relative_rms_change.median()),**{f"median_{b}_retention":float(g[f"{b}_retention"].median()) for b in BANDS}})
 summary=pd.DataFrame(summaries); base_leak=float(summary.loc[summary.strategy.eq("no_removal"),"median_max_aux_correlation"].iloc[0]); phys=summary.loc[summary.strategy.eq("physiological_only")].iloc[0]
 band_ok=all(.75<=phys[f"median_{b}_retention"]<=1.25 for b in BANDS); rms_ok=phys.median_relative_rms_change<=.20; improvement=(base_leak-phys.median_max_aux_correlation)/max(base_leak,1e-20); selected=physiological if physiological and improvement>=.10 and band_ok and rms_ok else []
 decision={"checkpoint":"P6b","subject":a.subject,"bad_channels":bads,"review_candidates":candidates,"physiological_candidates":physiological,"selected_components":selected,"guardrails":{"minimum_aux_leakage_reduction":.10,"maximum_relative_rms_change":.20,"band_retention_range":[.75,1.25]},"observed":{"aux_leakage_reduction":float(improvement),"band_guardrail_pass":bool(band_ok),"rms_guardrail_pass":bool(rms_ok)},"decision":"remove_conservative_physiological_components" if selected else "no_ica_component_removed","component_indices_are_subject_specific":True}
 detail.to_csv(out/"p6b_ica_ablation_by_trial.csv",index=False); summary.to_csv(out/"p6b_ica_ablation_summary.csv",index=False); (out/"p6b_ica_decision.json").write_text(json.dumps(decision,indent=2)+"\n",encoding="utf-8"); print(json.dumps(decision,indent=2))
if __name__=="__main__": main()
