"""Run the approved subject-specific DENS pipeline with resumable logging."""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import pandas as pd

STEPS = [
 ("bad_channels", "detect_dens_bad_channels.py", lambda s: ["--dataset-root","data/raw/ds003751","--manifest","data/manifest/dens_trials.csv","--subject",s], "p4_bad_channel_summary.json"),
 ("ica_fit", "propose_dens_ica_components.py", lambda s: ["--dataset-root","data/raw/ds003751","--manifest","data/manifest/dens_trials.csv","--subject",s], "p6_ica_summary.json"),
 ("ica_ablation", "ablate_dens_ica_subject.py", lambda s: ["--dataset-root","data/raw/ds003751","--manifest","data/manifest/dens_trials.csv","--ica",f"reports/qc/{s}/p6_ica_solution-ica.fif","--subject",s], "p6b_ica_decision.json"),
 ("trial_processing", "process_and_qc_dens_trials.py", lambda s: ["--dataset-root","data/raw/ds003751","--manifest","data/manifest/dens_trials.csv","--ica",f"reports/qc/{s}/p6_ica_solution-ica.fif","--subject",s], "p7_p8_summary.json"),
 ("window_qc", "qc_dens_windows.py", lambda s: ["--subject",s], "p9_summary.json"),
 ("curation", "curate_dens_subject.py", lambda s: ["--subject",s], "p10_curation_summary.json"),
]

def materialized(root: Path, subject: str) -> bool:
 files=list((root/subject/"eeg").glob("*_eeg.set"))+list((root/subject/"eeg").glob("*_eeg.fdt"))
 return len(files)==2 and all(f.stat().st_size>1024 for f in files)

def main():
 p=argparse.ArgumentParser(); p.add_argument("--subjects",nargs="*"); p.add_argument("--available-only",action="store_true"); p.add_argument("--force",action="store_true"); a=p.parse_args()
 root=Path("data/raw/ds003751"); table=pd.read_csv("data/manifest/dens_subjects.csv")
 eligible=table[table.has_subject_directory&table.has_eeg_metadata&table.has_events&table.has_behavior].subject.tolist()
 subjects=a.subjects or [s for s in eligible if s!="sub-mit003"]
 env=os.environ.copy(); cache=Path(".cache").resolve(); (cache/"numba").mkdir(parents=True,exist_ok=True); (cache/"tmp").mkdir(parents=True,exist_ok=True); env.update({"NUMBA_CACHE_DIR":str(cache/"numba"),"TEMP":str(cache/"tmp"),"TMP":str(cache/"tmp")})
 log_path=Path("reports/cohort_processing_status.csv"); rows=[]
 for subject in subjects:
  if not materialized(root,subject):
   rows.append({"subject":subject,"step":"raw","status":"pending_download","seconds":0,"error":""})
   if a.available_only: continue
   raise RuntimeError(f"Raw EEG is not materialized for {subject}")
  failed=False
  for step,script,args_fn,artifact in STEPS:
   target=Path("reports/qc")/subject/artifact
   if target.exists() and not a.force:
    rows.append({"subject":subject,"step":step,"status":"already_complete","seconds":0,"error":""}); continue
   start=time.perf_counter(); command=[sys.executable,str(Path("scripts")/script),*args_fn(subject)]
   print("RUN",subject,step,flush=True)
   try:
    subprocess.run(command,check=True,env=env); status="complete"; error=""
   except subprocess.CalledProcessError as exc:
    status="failed"; error=f"exit_code={exc.returncode}"; failed=True
   rows.append({"subject":subject,"step":step,"status":status,"seconds":round(time.perf_counter()-start,2),"error":error})
   pd.DataFrame(rows).to_csv(log_path,index=False)
   if failed: break
  print("DONE" if not failed else "FAILED",subject,flush=True)
 pd.DataFrame(rows).to_csv(log_path,index=False)
 summary={"eligible_subjects":len(eligible),"requested":len(subjects),"fully_curated":int(sum((Path("reports/qc")/s/"p10_curation_summary.json").exists() for s in subjects)),"failed_subjects":sorted(set(r["subject"] for r in rows if r["status"]=="failed")),"pending_download":sorted(set(r["subject"] for r in rows if r["status"]=="pending_download"))}
 Path("reports/cohort_processing_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8"); print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
