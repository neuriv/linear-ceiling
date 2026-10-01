"""Verify local records, compare the bridge, and aggregate handoffs without pooling replicas."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import tomllib
import numpy as np
from linear_ceiling.e9_pertoken import centered_delta, token_mean, f_star, seam_distance_left
from linear_ceiling.rng import make_rng

ROOT = Path(__file__).resolve().parents[2]
def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence",type=Path,default=Path.home()/"Desktop/Carryover-evidence")
    parser.add_argument("--bridge-only",action="store_true")
    args = parser.parse_args()
    cfg = tomllib.loads((ROOT/"config/consolidation.toml").read_text())
    manifest = json.loads((ROOT/"config/consolidation-manifest.json").read_text())
    models = ["qwen17_bridge"] if args.bridge_only else list(manifest["models"])
    rows, summaries, bridge = [], {}, []
    for model in models:
        folder = args.evidence/"a100/results"/model
        report = json.loads((folder/"report.json").read_text())
        assert report["complete"],model
        assert report["config_sha256"]==digest(ROOT/"config/consolidation.toml")
        assert report["manifest_sha256"]==digest(ROOT/"config/consolidation-manifest.json")
        expected = {r["handoff"] for r in manifest["models"][model] if not r["excluded"]}
        assert set(report["scores"])==expected
        group_rows = []
        for hid,entry in report["scores"].items():
            assert digest(folder/entry["file"])==entry["sha256"]
            for file,sha in entry.get("witness",{}).items():
                assert digest(folder/file)==sha,file
            input_entry = next(r for r in manifest["models"][model] if r["handoff"]==hid)
            ipath = args.evidence/"a100/inputs"/input_entry["file"]
            assert digest(ipath)==input_entry["sha256"]
            with np.load(ipath) as x:
                distances = seam_distance_left(x["pairs"],len(x["receiver"]))
                n = len(x["pairs"])
            with np.load(folder/entry["file"]) as data:
                for kind in ["K","V"]:
                    squares,sst = data[kind+"_squares"],data[kind+"_sst"]
                    d = token_mean(centered_delta(squares,sst,n))
                    np.testing.assert_allclose(d,data[kind+"_token_delta"],rtol=1e-12,atol=1e-14)
                    mean = float(d.mean())
                    np.testing.assert_allclose(mean,1-(1-squares.sum(0,dtype=np.float64)/sst).mean(),rtol=1e-12)
                    measured = entry["metrics"][kind]
                    np.testing.assert_allclose(mean,measured["mean_delta"],rtol=1e-12)
                    tau = cfg[f"tau_{kind}"]
                    assert f_star(d,tau)==measured["fstar"]
                    if entry.get("witness"):
                        source = np.load(folder/"witness/sender"/f"{kind}.npy",mmap_mode="r")
                        fresh = np.load(folder/"witness/receiver"/f"{kind}.npy",mmap_mode="r")
                        for layer in range(len(source)):
                            ref = fresh[layer].astype(np.float64)
                            raw_sq = ((source[layer].astype(np.float64)-ref)**2).sum(-1).astype(np.float32)
                            np.testing.assert_array_equal(raw_sq,squares[:,layer])
                            np.testing.assert_allclose(((ref-ref.mean(0))**2).sum((0,2)),sst[layer],rtol=1e-12)
                    row = {"model":model,"handoff":hid,"trajectory":hid.split("#")[0],"cohort":entry["cohort"],
                           "kind":kind,"sender_tokens":entry["sender_tokens"],"receiver_tokens":entry["receiver_tokens"],
                           "matched_tokens":n,"mean_delta":mean,"r2":1-mean,"tail_fraction":float((d>tau).mean()),
                           "fstar":f_star(d,tau),"fstar_0.03":f_star(d,.03),"fstar_0.1":f_star(d,.1),
                           "near_mean":float(d[distances<16].mean()) if (distances<16).any() else None,
                           "far_mean":float(d[distances>=16].mean()) if (distances>=16).any() else None}
                    group_rows.append(row)
                    if model=="qwen17_bridge":
                        old = args.evidence/"archive/records"/entry["cohort"]
                        old_report = json.loads((old/"report.json").read_text())
                        rec = old_report["scores"][hid]
                        original = json.loads((old/"scores"/rec["score_file"]).read_text())
                        with np.load(old/"tokens"/rec["tokens_file"]) as x:
                            old_sst = np.array([l["sst"] for l in original["same"][kind]])
                            old_d = token_mean(centered_delta(x[f"same_{kind}"],old_sst,n))
                        mean_gap = abs(mean-old_d.mean())/old_d.mean()
                        fs_gap = max(abs(f_star(d,t)-f_star(old_d,t)) for t in [tau,.03,.1])
                        bridge.append({"handoff":hid,"kind":kind,"relative_mean_gap":float(mean_gap),"max_fstar_gap":fs_gap})
                        assert mean_gap<=cfg["bridge_mean_relative_tolerance"],bridge[-1]
                        assert fs_gap<=cfg["bridge_fstar_absolute_tolerance"],bridge[-1]
        rows.extend(group_rows)
        summaries[model] = {}
        for cohort in ["e9s","e9l"]:
            for kind in ["K","V"]:
                group = [r for r in group_rows if r["cohort"]==cohort and r["kind"]==kind]
                if not group:
                    continue
                metrics = ["mean_delta","r2","tail_fraction","fstar","fstar_0.03","fstar_0.1","near_mean","far_mean"]
                trajectories = sorted({r["trajectory"] for r in group})
                cluster = {t:[r for r in group if r["trajectory"]==t] for t in trajectories}
                rng = make_rng(cfg["seed"])
                boot = {m:[] for m in metrics}
                for sample in rng.integers(0,len(trajectories),size=(2000,len(trajectories))):
                    sampled = [r for i in sample for r in cluster[trajectories[i]]]
                    for m in metrics:
                        boot[m].append(float(np.median([r[m] for r in sampled if r[m] is not None])))
                summaries[model][cohort+"_"+kind] = {"handoffs":len(group),"trajectories":len(trajectories),
                    "sender_min_max":[min(r['sender_tokens'] for r in group),max(r['sender_tokens'] for r in group)],
                    "receiver_min_max":[min(r['receiver_tokens'] for r in group),max(r['receiver_tokens'] for r in group)],
                    "zero_fstar_handoffs":sum(r["fstar"]==0 for r in group),
                    "median":{m:float(np.median([r[m] for r in group if r[m] is not None])) for m in metrics},
                    "cluster_bootstrap_95":{m:np.quantile(boot[m],[.025,.975]).tolist() for m in metrics}}
    output = args.evidence/"a100"
    prefix = "bridge" if args.bridge_only else "summary"
    result = {"verified":True,"config_sha256":digest(ROOT/"config/consolidation.toml"),
              "models":summaries,"bridge":bridge,"bootstrap":"2000 trajectory-cluster resamples; handoff median; seed from config"}
    (output/f"{prefix}.json").write_text(json.dumps(result,indent=2)+"\n")
    with (output/f"{prefix}.csv").open("w") as f:
        w = csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(json.dumps({"verified":True,"bridge_max_relative_mean_gap":max(r['relative_mean_gap'] for r in bridge),
                      "bridge_max_fstar_gap":max(r['max_fstar_gap'] for r in bridge),"models":summaries},indent=2))

if __name__=="__main__":
    main()
