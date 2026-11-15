#!/usr/bin/env bash
# Runs on the Azure Linux VM (operon-linux), as root, through `az vm run-command` from
# .github/workflows/osworld-bench.yml (KAN-169). Do not run it on your PC.
#
# Input arrives as environment variables from the workflow:
#   GEMINI_API_KEY (protected parameter, never echoed), GH_TOKEN (contents: read, short-lived),
#   REPO, REF, LIMIT, ONLY, MAX_COST_USD, MAX_STEPS.
# The VM must already have ~/osw-venv (Python 3.11 with OSWorld's requirements),
# ~/operon-venv (Python 3.14), ~/OSWorld (a checkout of xlang-ai/OSWorld) and Docker, as set up in
# docs/osworld-ci.md. The output of run-command keeps only its last 4 KB, so this prints one line
# per task plus a total; the full report stays on the VM in .var/bench/osworld-*.json.
set -euo pipefail

U=azureuser
H=/home/$U
SRC=$H/operon-ci

: "${GEMINI_API_KEY:?missing}" "${REPO:?missing}" "${REF:?missing}"

# Fresh checkout of the ref to benchmark. The token goes in an HTTP header for this one fetch
# and is not stored in the repo's config.
rm -rf "$SRC"
sudo -u "$U" -H git init -q "$SRC"
sudo -u "$U" -H git -C "$SRC" -c "http.extraheader=Authorization: Bearer ${GH_TOKEN:-}" \
  fetch -q --depth 1 "https://github.com/${REPO}.git" "$REF"
sudo -u "$U" -H git -C "$SRC" checkout -q FETCH_HEAD

sudo -u "$U" -H "$H/osw-venv/bin/uv" pip install -q --python "$H/operon-venv/bin/python" -e "$SRC"

ARGS=(--max-cost-usd "${MAX_COST_USD:-2}" --max-steps "${MAX_STEPS:-50}"
      --operon-python "$H/operon-venv/bin/python")
[ -n "${ONLY:-}" ] && ARGS+=(--only "$ONLY") || ARGS+=(--limit "${LIMIT:-5}")

# The key reaches the runner only through its environment.
sudo -u "$U" -H env "GEMINI_API_KEY=$GEMINI_API_KEY" OSWORLD_REPO="$H/OSWorld" \
  OPERON_DESKTOP_PLATFORM=linux bash -c "cd '$SRC' && '$H/osw-venv/bin/python' -m evals.osworld $(printf '%q ' "${ARGS[@]}")" \
  > "$H/osworld-ci.log" 2>&1 || true

# Compact summary (4 KB limit): one line per task from the newest report.
REPORT=$(ls -t "$SRC"/.var/bench/osworld-*.json 2>/dev/null | head -1 || true)
if [ -z "$REPORT" ]; then
  echo "NO REPORT. Last log lines:"
  tr '\r' '\n' < "$H/osworld-ci.log" | grep -av 'Progress' | tail -15
  exit 1
fi
"$H/osw-venv/bin/python" - "$REPORT" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
s = d["summary"]
for r in d["results"]:
    print("%s %s %s $%.3f %ss %s" % ("PASS" if r["passed"] else "FAIL", r["domain"], r["id"][:8],
                                     r["cost_usd"], r["seconds"], r.get("note", "")))
print("TOTAL %d/%d passed, $%.3f" % (s["passed"], s["tasks"], s["cost_usd"]))
EOF
# No OSWorld container may outlive the run.
docker ps -q | xargs -r docker stop >/dev/null 2>&1 || true
