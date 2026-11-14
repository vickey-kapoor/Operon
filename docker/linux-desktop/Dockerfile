# Operon's Linux desktop container (KAN-80, stage 3): an XFCE desktop on a virtual
# display (Xvfb :1) with Operon's guest agent, which the host drives over HTTP.
# Built by the `linux-container` environment provider from a source checkout:
#   docker build -f docker/linux-desktop/Dockerfile -t operon-linux-desktop .
# The Gemini key never goes in: the host makes every model call.
FROM python:3.14-slim-trixie

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      xvfb xauth dbus-x11 xfce4 xfce4-terminal \
      mousepad galculator thunar ristretto \
      chromium firefox-esr \
      xclip fonts-dejavu-core libatomic1 procps \
      libtk8.6 \
 && rm -rf /var/lib/apt/lists/*

# Operon and its dependencies. At run time the provider mounts the host's current
# source over /opt/operon-src (read-only), so code changes need no rebuild.
COPY pyproject.toml README.md LICENSE /opt/operon/
COPY src /opt/operon/src
# Behind a TLS-inspecting proxy, pass its CA bundle: --secret id=ca_bundle,src=<file>.
RUN --mount=type=secret,id=ca_bundle,required=false \
    if [ -f /run/secrets/ca_bundle ]; then export PIP_CERT=/run/secrets/ca_bundle; fi \
 && pip install --no-cache-dir /opt/operon && mkdir -p /opt/operon-src

RUN useradd --create-home --shell /bin/bash operon \
 && touch /home/operon/.Xauthority \
 && chown operon:operon /home/operon/.Xauthority
COPY docker/linux-desktop/entrypoint.sh /usr/local/bin/operon-desktop
RUN chmod 0755 /usr/local/bin/operon-desktop

USER operon
WORKDIR /home/operon
ENV DISPLAY=:1 \
    HOME=/home/operon \
    OPERON_DESKTOP_PLATFORM=linux \
    OPERON_RUNTIME_ROOT=/home/operon/.operon-guest \
    OPERON_CONFIG_DIR=/home/operon/.operon-guest \
    PYTHONDONTWRITEBYTECODE=1 \
    OPERON_SCREEN=1280x800x24
EXPOSE 8765
ENTRYPOINT ["/usr/local/bin/operon-desktop"]
