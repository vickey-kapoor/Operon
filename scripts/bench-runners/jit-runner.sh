#!/usr/bin/env bash
# One-job (just-in-time) runners for the Azure benchmark VMs (KAN-195).
#
# The repo is public, so a lasting self-hosted runner could be handed any approved fork-PR workflow
# that asks for its label. Instead nothing lasting stays registered: each paid run mints a JIT runner
# under a label unique to that run (with a random suffix), which takes exactly one job and is deleted
# by GitHub when the job ends.
#
#   bash jit-runner.sh start <linux|windows> <label> <static-label>
#       mint the JIT config, delete any old lasting runner that carries <static-label>, hand the
#       config to the VM (az vm run-command), then wait until the runner is online. Prints only the
#       VM script's output.
#   bash jit-runner.sh remove <label>
#       delete the runner named <label> if it is still registered (the job never started).
#
# Needs: GH_ADMIN_TOKEN (fine-grained token, this repo only, Administration read+write), GH_REPO
# (owner/name), and for `start` RESOURCE_GROUP, VM_NAME and an `az` that is already signed in.
# RUNNER_ONLINE_TIMEOUT / RUNNER_POLL_INTERVAL (seconds, default 300 / 10) tune the wait.
# The token and the config are secrets: never echo them, never put them in a command line.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${GH_REPO:?GH_REPO (owner/name) is required}"
: "${GH_ADMIN_TOKEN:?GH_ADMIN_TOKEN is required}"

# A private scratch dir (global: the EXIT trap runs after the functions have returned). The token
# goes to curl through a header file in it, so it is not in curl's command line.
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
chmod 700 "$tmp"
( umask 077; printf 'Authorization: Bearer %s\n' "$GH_ADMIN_TOKEN" > "$tmp/headers" )

# One GitHub API call: api METHOD PATH [JSON body]. -f makes an HTTP error a non-zero exit.
api() {
  local method="$1" path="$2" body="${3:-}"
  local args=(-fsS -X "$method" -H "@$tmp/headers"
              -H "Accept: application/vnd.github+json" -H "X-GitHub-Api-Version: 2022-11-28")
  if [ -n "$body" ]; then args+=(-H "Content-Type: application/json" -d "$body"); fi
  curl "${args[@]}" "https://api.github.com/repos/$GH_REPO/$path"
}

# Delete every registered runner for which the jq filter $1 (given the runner as `.`) is true.
delete_runners_where() {
  local filter="$1" ids id
  ids="$(api GET "actions/runners?per_page=100" | jq -r ".runners[] | select($filter) | .id")"
  for id in $ids; do
    api DELETE "actions/runners/$id" >/dev/null || echo "::warning::could not delete runner $id (busy or already gone)"
  done
}

# Wait until the runner named $1 is online, so a runner that never came up fails here (and stop-vm
# deallocates the VM at once) instead of the bench job queueing for hours.
wait_online() {
  local label="$1" waited=0 limit="${RUNNER_ONLINE_TIMEOUT:-300}" step="${RUNNER_POLL_INTERVAL:-10}"
  while :; do
    if [ -n "$(api GET "actions/runners?per_page=100" | jq -r ".runners[] | select(.name == \"$label\" and .status == \"online\") | .id")" ]; then
      echo "runner $label is online"
      return 0
    fi
    if [ "$waited" -ge "$limit" ]; then
      echo "::error::runner $label did not come online within ${limit}s" >&2
      return 1
    fi
    sleep "$step"
    waited=$((waited + step))
  done
}

check_label() {
  case "$1" in ("" | *[!a-z0-9-]*) echo "::error::runner label must be lowercase letters, digits and dashes" >&2; exit 1;; esac
}

start() {
  local os="$1" label="$2" static="$3" command_id template config out
  check_label "$label"; check_label "$static"
  : "${RESOURCE_GROUP:?RESOURCE_GROUP is required}" "${VM_NAME:?VM_NAME is required}"
  case "$os" in
    linux)   command_id=RunShellScript;      template="$HERE/vm-linux.sh" ;;
    windows) command_id=RunPowerShellScript; template="$HERE/vm-windows.ps1" ;;
    *) echo "::error::os must be linux or windows" >&2; exit 1 ;;
  esac

  # 1. The JIT config: base64, valid for one job, tied to the unique label (runner group 1 = Default).
  config="$(api POST "actions/runners/generate-jitconfig" \
    "$(jq -nc --arg name "$label" '{name: $name, runner_group_id: 1, labels: [$name], work_folder: "_work"}')" \
    | jq -r .encoded_jit_config)"
  echo "::add-mask::$config"
  # Base64 only, so it is safe inside single quotes in the script below.
  case "$config" in ("" | null | *[!A-Za-z0-9+/=]*) echo "::error::GitHub did not return a usable JIT config" >&2; exit 1;; esac

  # 2. The lasting runner of earlier setups must not stay registered under the old label.
  delete_runners_where "(.labels | map(.name) | index(\"$static\")) != null"

  # 3. The VM's script is the template with the config and label put in front of it.
  if [ "$os" = linux ]; then
    printf "JIT_CONFIG='%s'\nRUNNER_LABEL='%s'\n" "$config" "$label" > "$tmp/script"
  else
    printf "\$JitConfig = '%s'\n\$RunnerLabel = '%s'\n" "$config" "$label" > "$tmp/script"
  fi
  cat "$template" >> "$tmp/script"

  # az prints the script's output; the VM script ends with "operon-jit-runner-started" when it worked.
  out="$(az vm run-command invoke --resource-group "$RESOURCE_GROUP" --name "$VM_NAME" \
    --command-id "$command_id" --scripts "@$tmp/script" --query "value[].message" -o tsv)"
  echo "$out"
  case "$out" in (*operon-jit-runner-started*) ;; (*) echo "::error::the VM did not start the runner" >&2; exit 1;; esac

  # 4. Fail now if the runner does not come online.
  wait_online "$label"
}

remove() {
  check_label "$1"
  delete_runners_where ".name == \"$1\""
}

case "${1:-}" in
  start) [ $# -eq 4 ] || { echo "usage: $0 start <linux|windows> <label> <static-label>" >&2; exit 2; }; start "$2" "$3" "$4" ;;
  remove) [ $# -eq 2 ] || { echo "usage: $0 remove <label>" >&2; exit 2; }; remove "$2" ;;
  *) echo "usage: $0 start <linux|windows> <label> <static-label> | remove <label>" >&2; exit 2 ;;
esac
