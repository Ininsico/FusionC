import sys
from pathlib import Path

path = Path(sys.argv[1])
kernel = sys.argv[2]
lines = path.read_text(encoding="utf-8").splitlines()
start = next(i for i, l in enumerate(lines) if f"void {kernel}(" in l)
depth = 0
end = start
for i in range(start, len(lines)):
    depth += lines[i].count("{") - lines[i].count("}")
    if depth == 0 and i > start:
        end = i
        break
for i in range(max(0, start - 5), end + 1):
    print(f"{i + 1:4d}  {lines[i]}")
