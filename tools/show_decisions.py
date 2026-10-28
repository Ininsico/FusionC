import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
d = json.loads(path.read_text(encoding="utf-8"))
print("mode:", d["mode"], "smCount:", d["smCount"], "irNodes:", d["irNodes"])
for r in d["regions"]:
    print("  region %-22s kernel=%-22s kind=%-18s members=%s"
          % (r["region"], r["kernel"], r["kind"], r["members"]))
print()
for dec in d["decisions"]:
    tag = "FUSED " if dec["accepted"] else "REJECT"
    print(tag, dec["kind"], "->", dec["members"])
    pred = dec.get("predicted", {})
    if pred:
        print("        predicted:", pred)
    print("        reason:", (dec["reason"][:260]).strip())
