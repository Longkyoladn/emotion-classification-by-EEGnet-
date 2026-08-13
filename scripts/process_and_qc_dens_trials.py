"""P7/P8: create cleaned pilot trials and assign auditable QC states."""

from __future__ import annotations
import argparse, json
from pathlib import Path
import mne, numpy as np, pandas as pd
from scipy.signal import welch

EEG=[f"E{i}" for i in range(1,129)]; BADS=["E50","E103"]

def main():
 p=argparse.ArgumentParser(); p.add_argument("--dataset-root",type=Path,required=True); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--ica",type=Path,required=True); p.add_argument("--subject",default="sub-mit003"); p.add_argument("--processed-dir",type=Path,default=Path("data/processed")); p.add_argument("--qc-dir",type=Path,default=Path("reports/qc")); a=p.parse_args()
 mf=pd.read_csv(a.manifest); ts=mf[mf.subject.eq(a.subject)&mf.match_status.eq("matched")&mf.validation_issues.fillna("").eq("")].sort_values("event_trial_index")
 raw=mne.io.read_raw_eeglab(a.dataset_root/ts.eeg_set_path.dropna().unique()[0],preload=False,verbose="ERROR"); raw.set_channel_types({**{c:"eeg" for c in EEG},"E129":"misc","ECG":"ecg","EMG":"emg","EMG_2":"emg"},verbose="ERROR"); raw.set_montage(mne.channels.make_standard_montage("GSN-HydroCel-128"),on_missing="ignore",verbose="ERROR")
 ica=mne.preprocessing.read_ica(a.ica,verbose="ERROR"); sf=float(raw.info["sfreq"]); pad=int(15*sf); rows=[]; out=a.processed_dir/a.subject/"trials"; out.mkdir(parents=True,exist_ok=True)
 for tr in ts.itertuples(index=False):
  start,stop=int(round(tr.onset_seconds*sf)),int(round(tr.end_seconds*sf)); ps,pe=max(0,start-pad),min(raw.n_times,stop+pad)
  seg=raw.copy().crop(ps/sf,(pe-1)/sf,include_tmax=True).load_data(verbose="ERROR"); seg.notch_filter([50.],phase="zero",verbose="ERROR").filter(.5,45.,phase="zero",verbose="ERROR"); seg.crop((start-ps)/sf,(stop-ps-1)/sf,include_tmax=True)
  ecg=seg.get_data(picks=["ECG"])[0]; eeg=seg.copy().pick(EEG); eeg.info["bads"]=BADS; eeg.interpolate_bads(reset_bads=False,method={"eeg":"spline"},verbose="ERROR"); eeg.set_eeg_reference("average",projection=False,verbose="ERROR"); ica.apply(eeg,exclude=[11],verbose="ERROR")
  x=eeg.get_data()*1e6; finite=np.isfinite(x); std=np.std(x,axis=1); med=np.median(np.log10(std+1e-12)); mad=np.median(np.abs(np.log10(std+1e-12)-med)); z=(np.log10(std+1e-12)-med)/max(1.4826*mad,1e-12)
  f,pw=welch(x,fs=sf,nperseg=int(4*sf),noverlap=int(2*sf),axis=1); line=np.trapezoid(pw[:,(f>=49)&(f<51)],f[(f>=49)&(f<51)],axis=1); near=np.trapezoid(pw[:,(f>=45)&(f<49)],f[(f>=45)&(f<49)],axis=1)+np.trapezoid(pw[:,(f>=51)&(f<55)],f[(f>=51)&(f<55)],axis=1); line_ratio=np.median(line/np.maximum(near/4,1e-20)); ecg_corr=max(abs(np.corrcoef(xi,ecg)[0,1]) for xi in x); extreme_fraction=float(np.mean(np.abs(x)>200)); flat_count=int(np.sum(std<.5)); outlier_count=int(np.sum(np.abs(z)>3.5)); peak=float(np.max(np.abs(x)))
  issues=[]; severe=[]
  if not finite.all(): severe.append("nonfinite")
  if x.shape[0]!=128: severe.append("wrong_channel_count")
  if x.shape[1] < int(58*sf): severe.append("too_short")
  if peak>1000 or extreme_fraction>.01: severe.append("extreme_amplitude")
  if flat_count>0: issues.append("flat_channels")
  if outlier_count>5: issues.append("variance_outliers")
  if line_ratio>5: issues.append("residual_line_noise")
  if ecg_corr>.30: issues.append("residual_ecg")
  if peak>500 or extreme_fraction>.001: issues.append("amplitude_warning")
  status="reject" if severe else ("review" if issues else "pass")
  path=out/f"{a.subject}_trial-{int(tr.event_trial_index):02d}_eeg.fif"; eeg.save(path,overwrite=True,verbose="ERROR")
  rows.append({"subject":a.subject,"trial":int(tr.event_trial_index),"stimulus":tr.stimulus,"status":status,"issues":";".join(severe+issues),"samples":x.shape[1],"duration_seconds":x.shape[1]/sf,"channels":x.shape[0],"max_abs_uv":peak,"extreme_over_200uv_fraction":extreme_fraction,"flat_channels":flat_count,"variance_outlier_channels":outlier_count,"residual_50hz_ratio":line_ratio,"max_abs_ecg_correlation":ecg_corr,"interpolated_channels":";".join(BADS),"ica_removed":"11","output_path":path.as_posix()})
 q=pd.DataFrame(rows); qo=a.qc_dir/a.subject; qo.mkdir(parents=True,exist_ok=True); q.to_csv(qo/"p7_p8_trial_qc.csv",index=False)
 summary={"checkpoint":"P7_P8","subject":a.subject,"trials":len(q),"status_counts":q.status.value_counts().to_dict(),"pipeline":["notch 50 Hz","band-pass 0.5-45 Hz","interpolate E50/E103","average reference excluding bads","remove ICA IC11"],"thresholds":{"reject_peak_uv":1000,"reject_extreme_over_200uv_fraction":.01,"review_peak_uv":500,"review_extreme_fraction":.001,"review_residual_50hz_ratio":5,"review_ecg_correlation":.30,"review_variance_outlier_channels":5},"no_trials_deleted":True,"labels_derived":False}
 (qo/"p7_p8_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8"); print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
