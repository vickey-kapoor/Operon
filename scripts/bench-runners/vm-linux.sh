# shellcheck shell=bash
# Runs on operon-linux as root (az vm run-command, RunShellScript), started by jit-runner.sh (KAN-195).
# jit-runner.sh puts two lines in front of this file:
#   JIT_CONFIG='<base64 one-job runner config>'
#   RUNNER_LABEL='<operon-osworld-RUNID-ATTEMPT-RANDOM>'
# It starts the one-job runner detached (run-command must not wait for the job) and exits.
set -eu

# The unpacked runner (docs/osworld-runner.md): RUNNER_DIR if set, else the first ~/actions-runner.
RUNNER_DIR="${RUNNER_DIR:-}"
if [ -z "$RUNNER_DIR" ]; then
  for f in /home/*/actions-runner/run.sh; do
    if [ -e "$f" ]; then RUNNER_DIR="$(dirname "$f")"; break; fi
  done
fi
if [ -z "$RUNNER_DIR" ] || [ ! -x "$RUNNER_DIR/run.sh" ]; then
  echo "No unpacked runner found (expected ~/actions-runner/run.sh)." >&2
  exit 1
fi
RUNNER_USER="$(stat -c %U "$RUNNER_DIR")"
RUNNER_HOME="$(getent passwd "$RUNNER_USER" | cut -d: -f6)"
cd "$RUNNER_DIR"

# Move off the old lasting runner (safe to repeat): stop and remove its service, then drop its
# registration files, because --jitconfig refuses a folder that is already configured.
if [ -f .runner ]; then
  ./svc.sh stop || true
  ./svc.sh uninstall || true
fi
rm -f .runner .credentials .credentials_rsaparams
# A one-job runner from an earlier run that never got its job.
systemctl stop 'operon-jit-runner-*' 2>/dev/null || true
systemctl reset-failed 'operon-jit-runner-*' 2>/dev/null || true

# The config goes to a file in a directory root creates (0700) and the runner user owns, so the unit can
# read and delete it: it does both first thing, so the config is never in a command line or the
# environment and does not stay on disk.
CONFIG_DIR=/run/operon-jit
mkdir -p "$CONFIG_DIR"
chown "$RUNNER_USER" "$CONFIG_DIR"
chmod 700 "$CONFIG_DIR"
CONFIG_FILE="$(mktemp "$CONFIG_DIR/config.XXXXXX")"
printf '%s' "$JIT_CONFIG" > "$CONFIG_FILE"
chown "$RUNNER_USER" "$CONFIG_FILE"
chmod 600 "$CONFIG_FILE"

# A transient unit as the runner user (systemd gives it the user's groups, docker and kvm, which
# setup.sh needs). Unlike a child of this script it survives run-command ending.
# shellcheck disable=SC2016  # the single quotes are on purpose: $1 and $cfg expand in the unit's shell
systemd-run --quiet --collect --unit="operon-jit-runner-$RUNNER_LABEL" --uid="$RUNNER_USER" \
  -p WorkingDirectory="$RUNNER_DIR" --setenv=HOME="$RUNNER_HOME" \
  --setenv=USER="$RUNNER_USER" --setenv=LOGNAME="$RUNNER_USER" \
  /bin/bash -c 'cfg=$(cat "$1"); rm -f "$1"; exec ./run.sh --jitconfig "$cfg"' _ "$CONFIG_FILE"

echo "operon-jit-runner-started $RUNNER_LABEL"
