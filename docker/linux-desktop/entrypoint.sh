#!/bin/sh
# Start the virtual display and the XFCE desktop, then Operon's guest agent in the
# foreground (the container stops when it does). OPERON_GUEST_TOKEN comes from the host.
set -e
Xvfb :1 -screen 0 "${OPERON_SCREEN}" -nolisten tcp >/tmp/xvfb.log 2>&1 &
for _ in $(seq 100); do
  [ -e /tmp/.X11-unix/X1 ] && break
  sleep 0.1
done
dbus-launch --exit-with-session startxfce4 >/tmp/xfce.log 2>&1 &
sleep 3
# The host's current source, when mounted, wins over the copy baked into the image.
if [ -d /opt/operon-src/operon ]; then
  export PYTHONPATH=/opt/operon-src
fi
exec python -m operon.cli guest --host 0.0.0.0 --port 8765
