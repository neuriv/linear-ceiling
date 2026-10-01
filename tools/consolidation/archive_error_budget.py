"""Does deleting all near-seam tokens suffice under the existing mean-error diagnostic?"""
from pathlib import Path
import hashlib
import json
import numpy as np
from linear_ceiling.e9_pertoken import centered_delta, token_mean, seam_distance_left

base = Path.home()/"Desktop/Carryover-evidence/archive/records"
results = {}
for cohort in ["e9s","e9l"]:
    folder = base/cohort
    report = json.loads((folder/"report.json").read_text())
    rows = []
    for hid,entry in report["scores"].items():
        token_file = folder/"tokens"/entry["tokens_file"]
        with token_file.open("rb") as f:
            assert hashlib.file_digest(f,"sha256").hexdigest()==entry["tokens_sha256"]
        score = json.loads((folder/"scores"/entry["score_file"]).read_text())
        with np.load(folder/"align"/(Path(entry["score_file"]).stem+".npz")) as a:
            far = seam_distance_left(a["pairs"],len(a["receiver"]))>=16
        with np.load(token_file) as a:
            sst = np.array([l["sst"] for l in score["same"]["K"]])
            d = token_mean(centered_delta(a["same_K"],sst,len(far)))
        assert far.any()
        rows.append({"handoff":hid,"far_token_share":float(far.mean()),
                     "far_error_share":float(d[far].sum()/d.sum()),"retained_far_mean":float(d[far].mean())})
    results[cohort] = {"handoffs":len(rows),"far_mean_above_0.03":sum(r["retained_far_mean"]>.03 for r in rows),
        "median":{k:float(np.median([r[k] for r in rows])) for k in ["far_token_share","far_error_share","retained_far_mean"]},
        "per_handoff":rows}
output = base.parent/"error-budget.json"
output.write_text(json.dumps(results,indent=2)+"\n")
print(json.dumps({k:{x:y for x,y in v.items() if x!="per_handoff"} for k,v in results.items()},indent=2))
