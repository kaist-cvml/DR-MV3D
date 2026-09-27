#!/usr/bin/env bash
# Everything that must pass before this repository is published.
#
#     bash tools/preflight.sh
#
# Linting, syntax, the test suite, that every module imports and every entry point
# runs, that the documented shell blocks are valid, and that nothing unpublishable
# is in the tree. Needs no data and no GPU.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON="${PYTHON:-python3}"
failed=()

step() {
  local name="$1"; shift
  printf '\n\033[1m== %s ==\033[0m\n' "$name"
  if "$@"; then
    return 0
  fi
  failed+=("$name")
  return 0
}

step "lint" "$PYTHON" -m ruff check .

step "python 3.10 syntax" "$PYTHON" - <<'PY'
import ast, pathlib, sys
checked = 0
for path in sorted(pathlib.Path(".").rglob("*.py")):
    if "cache" in str(path):
        continue
    try:
        ast.parse(path.read_text(), filename=str(path), feature_version=(3, 10))
    except SyntaxError as error:
        print(f"FAIL {path}:{error.lineno}: {error.msg}")
        sys.exit(1)
    checked += 1
print(f"{checked} files parse")
PY

step "tests" "$PYTHON" -m pytest tests/ -q

step "every module imports" "$PYTHON" - <<'PY'
import importlib, pathlib
for path in sorted(pathlib.Path("drmv3d").rglob("*.py")):
    importlib.import_module(str(path.with_suffix("")).replace("/", ".").removesuffix(".__init__"))
print("all modules import")
PY

step "every entry point runs" bash -c '
  shopt -s nullglob
  status=0
  count=0
  for script in scripts/*.py tools/*.py; do
    count=$((count + 1))
    "${PYTHON:-python3}" "$script" --help >/dev/null 2>&1 || { echo "FAIL $script"; status=1; }
  done
  for script in scripts/*.sh tools/*.sh; do
    count=$((count + 1))
    bash -n "$script" || { echo "FAIL $script"; status=1; }
  done
  [ $status -eq 0 ] && echo "$count entry points load"
  exit $status
'

step "documented commands are valid shell" "$PYTHON" - <<'PY'
import os, pathlib, re, subprocess, sys, tempfile
bad = 0
for doc in sorted(pathlib.Path(".").glob("*.md")) + sorted(pathlib.Path("docs").glob("*.md")):
    for index, block in enumerate(re.findall(r"```bash\n(.*?)```", doc.read_text(), re.DOTALL), 1):
        probe = re.sub(r"<[^>\s]+>", "PLACEHOLDER", block)
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as handle:
            handle.write(probe)
            path = handle.name
        result = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
        os.unlink(path)
        if result.returncode:
            print(f"FAIL {doc} block {index}: {result.stderr.strip()[:100]}")
            bad += 1
print("all documented commands parse" if not bad else f"{bad} blocks are invalid")
sys.exit(1 if bad else 0)
PY

step "release hygiene" bash tools/check_release_hygiene.sh

printf '\n%s\n' "------------------------------------------------------------"
if [[ ${#failed[@]} -eq 0 ]]; then
  echo "PASS: ready to publish"
  exit 0
fi
echo "FAILED: ${#failed[@]} check(s) did not pass"
for name in "${failed[@]}"; do
  echo "  - $name"
done
exit 1
