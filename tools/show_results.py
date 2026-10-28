import json
from pathlib import Path

data = json.loads(Path("results/results.json").read_text(encoding="utf-8"))
for w in data["workloads"]:
    print("workload:", w["workload"])
    for tag, v in w["variants"].items():
        print(f"  {tag:<22} compiled={v.get('compiled')} total_ms={v.get('total_ms')} "
              f"run_error={v.get('run_error')}")
        if v.get("nvcc_errors"):
            print("     nvcc:", v["nvcc_errors"][:2])
