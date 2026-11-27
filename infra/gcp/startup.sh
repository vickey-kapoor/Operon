#!/bin/bash
# Startup script of the OSWorld runner VM (KAN-224); Terraform puts this file on the VM's metadata.
# Google's guest agent runs it as root on every boot. It must be safe to cut short and run again:
# Terraform creates the VM and stops it at once, so the first boot rarely finishes. Every step checks
# what is already in place, and the services that act on their own (the runner watcher and the idle
# power-off) are started only at the end, never enabled, so they never run on a half-provisioned VM.
#
#   operon-memlog     every 30 s: free -m and the 5 biggest processes (names only) to the journal and
#                     the serial console (gcloud compute instances get-serial-port-output reads it)
#   operon-jit-watch  waits for the one-job runner script the workflow puts on the metadata key
#                     operon-jit-script (scripts/bench-runners/jit-runner.sh) and runs it once
#   operon-idle-off   powers the VM off after 20 minutes with no one-job runner (billing backstop)
#
# The GitHub Actions runner is unpacked in /home/runner/actions-runner, owned by the user `runner`
# (groups docker and kvm). It is never configured here: each run brings its own one-job config.
set -euo pipefail

RUNNER_VERSION=2.338.2
RUNNER_SHA256=baebf204307bac6193480852732f0aba2ae97b6c5597c244f4d5558390147c40
RUNNER_USER=runner
RUNNER_HOME=/home/runner
RUNNER_DIR="$RUNNER_HOME/actions-runner"
PACKAGES=(docker.io python3-venv python3-pip python3-dev build-essential git jq curl)

log() { echo "operon-startup: $*"; }

# write_file PATH MODE: the file's content comes on stdin.
write_file() {
  local tmp
  tmp="$(mktemp "$1.XXXXXX")"
  cat > "$tmp"
  chmod "$2" "$tmp"
  mv -f "$tmp" "$1"
}

# 1. The memory log first, so a slow or stuck provisioning shows up in it too.
write_file /usr/local/sbin/operon-memlog 0755 <<'EOF'
#!/bin/bash
# KAN-224: memory every 30 s, to the journal and the serial console (which survives a frozen VM).
# Process names only (ps comm), never their arguments: those can hold secrets.
while :; do
  { date -u +%FT%TZ; free -m; ps -eo rss,comm --sort=-rss | head -n 6; } 2>&1 \
    | sed 's/^/operon-memlog: /' | tee -a /dev/ttyS0 || true
  sleep 30
done
EOF
write_file /etc/systemd/system/operon-memlog.service 0644 <<'EOF'
[Unit]
Description=Operon memory log (KAN-224)

[Service]
ExecStart=/usr/local/sbin/operon-memlog
Restart=always
RestartSec=5
EOF
systemctl daemon-reload
systemctl restart operon-memlog.service

# 2. 8 GB of swap, used only under pressure. The fstab line is written last, so a cut-short run starts over.
if ! grep -q '^/swapfile ' /etc/fstab; then
  swapoff /swapfile 2>/dev/null || true
  rm -f /swapfile
  fallocate -l 8G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile >/dev/null
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi
swapon --show=NAME --noheadings | grep -qx /swapfile || swapon /swapfile
echo 'vm.swappiness=10' > /etc/sysctl.d/90-operon-swap.conf
sysctl -q -w vm.swappiness=10

# 3. Packages (setup.sh needs Docker, Python 3.11 venvs and git; jit-runner's VM script needs systemd-run).
export DEBIAN_FRONTEND=noninteractive
dpkg --configure -a
missing=0
for pkg in "${PACKAGES[@]}"; do
  # shellcheck disable=SC2016  # ${Status} is dpkg-query's field, not a shell variable
  dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q '^install ok installed$' || missing=1
done
if [ "$missing" = 1 ]; then
  log "installing packages"
  apt-get -o DPkg::Lock::Timeout=600 update -q
  apt-get -o DPkg::Lock::Timeout=600 install -y -q "${PACKAGES[@]}"
fi
systemctl enable --now docker.service

# 4. The runner user, with Docker and KVM (OSWorld's task VM).
getent group kvm >/dev/null || groupadd --system kvm
id -u "$RUNNER_USER" >/dev/null 2>&1 || useradd --create-home --shell /bin/bash "$RUNNER_USER"
usermod -aG docker,kvm "$RUNNER_USER"

# 5. The Actions runner, checked against its published sha256. Unpacked next to its final place and
#    moved in whole; the version file is written last.
if [ "$(cat "$RUNNER_DIR/.operon-runner-version" 2>/dev/null || true)" != "$RUNNER_VERSION" ]; then
  log "installing actions runner $RUNNER_VERSION"
  rm -rf "$RUNNER_HOME"/.runner-new.*
  work="$(mktemp -d "$RUNNER_HOME/.runner-new.XXXXXX")"
  curl -fsSL --retry 5 -o "$work/runner.tar.gz" \
    "https://github.com/actions/runner/releases/download/v$RUNNER_VERSION/actions-runner-linux-x64-$RUNNER_VERSION.tar.gz"
  echo "$RUNNER_SHA256  $work/runner.tar.gz" | sha256sum -c --quiet -
  mkdir "$work/runner"
  tar -xzf "$work/runner.tar.gz" -C "$work/runner"
  "$work/runner/bin/installdependencies.sh" >/dev/null
  rm -rf "$RUNNER_DIR"
  mv "$work/runner" "$RUNNER_DIR"
  rm -rf "$work"
  chown -R "$RUNNER_USER:$RUNNER_USER" "$RUNNER_DIR"
  echo "$RUNNER_VERSION" > "$RUNNER_DIR/.operon-runner-version"
fi

# 6. The runner watcher. The script on the metadata key holds the one-job config, so it is never
#    printed: it goes to a root-only file under /run, runs once, and is deleted.
write_file /usr/local/sbin/operon-jit-watch 0755 <<'EOF'
#!/bin/bash
# KAN-224: run each new one-job runner script from the metadata key operon-jit-script once.
set -u
umask 077
URL=http://metadata.google.internal/computeMetadata/v1/instance/attributes/operon-jit-script
STATE=/run/operon-jit-watch
mkdir -p "$STATE"
chmod 700 "$STATE"
etag=0
while :; do
  rm -f "$STATE/body" "$STATE/headers"
  code="$(curl -sS -o "$STATE/body" -D "$STATE/headers" -w '%{http_code}' -H 'Metadata-Flavor: Google' \
    "$URL?wait_for_change=true&timeout_sec=60&last_etag=$etag" 2>/dev/null)" || code=000
  if [ "$code" != 200 ] || [ ! -s "$STATE/body" ]; then
    etag=0
    sleep 5
    continue
  fi
  etag="$(sed -n 's/^[Ee][Tt][Aa][Gg]: *\([0-9a-zA-Z]*\).*/\1/p' "$STATE/headers" | head -n 1)"
  etag="${etag:-0}"
  sum="$(sha256sum < "$STATE/body" | cut -d' ' -f1)"
  if [ -e "$STATE/ran-$sum" ]; then
    continue
  fi
  touch "$STATE/ran-$sum"
  script="$(mktemp "$STATE/script.XXXXXX")"
  mv -f "$STATE/body" "$script"
  chmod 700 "$script"
  echo "operon-jit-watch: running runner script ${sum:0:12}"
  RUNNER_DIR=/home/runner/actions-runner bash "$script" || echo "operon-jit-watch: runner script ${sum:0:12} failed ($?)"
  rm -f "$script"
done
EOF
write_file /etc/systemd/system/operon-jit-watch.service 0644 <<'EOF'
[Unit]
Description=Operon one-job runner watcher (KAN-224)
After=network-online.target docker.service
Wants=network-online.target

[Service]
ExecStart=/usr/local/sbin/operon-jit-watch
Restart=always
RestartSec=5
EOF

# 7. The idle power-off. It waits until the watcher runs, so it never fires during provisioning.
write_file /usr/local/sbin/operon-idle-off 0755 <<'EOF'
#!/bin/bash
# KAN-224: billing backstop. Power off after 20 minutes in which no one-job runner unit was active.
set -u
STAMP=/run/operon-idle-off.busy
systemctl is-active --quiet operon-jit-watch.service || exit 0
if [ ! -e "$STAMP" ] || [ -n "$(systemctl list-units --type=service --state=active,activating --no-legend --plain 'operon-jit-runner-*')" ]; then
  touch "$STAMP"
  exit 0
fi
idle=$(( $(date +%s) - $(stat -c %Y "$STAMP") ))
if [ "$idle" -ge 1200 ]; then
  echo "operon-idle-off: no runner for ${idle}s, powering off" | tee -a /dev/ttyS0
  systemctl poweroff
fi
EOF
write_file /etc/systemd/system/operon-idle-off.service 0644 <<'EOF'
[Unit]
Description=Operon idle power-off check (KAN-224)

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/operon-idle-off
EOF
write_file /etc/systemd/system/operon-idle-off.timer 0644 <<'EOF'
[Unit]
Description=Operon idle power-off check every minute (KAN-224)

[Timer]
OnActiveSec=1min
OnUnitActiveSec=1min
AccuracySec=10s
EOF

systemctl daemon-reload
systemctl restart operon-jit-watch.service
systemctl restart operon-idle-off.timer
log "ready"
