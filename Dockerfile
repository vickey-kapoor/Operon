# =============================================================================
# UI Navigator — Multi-stage Dockerfile
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: builder — install Python dependencies into a virtual environment
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build tools required by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: runtime — lean image with Playwright Chromium
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# ── Copy virtual environment from builder ───────────────────────────────────
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ── System dependencies + Playwright browser binaries (single layer) ────────
# Keep apt cache available through the playwright install step so --with-deps
# can resolve additional system packages without needing a second apt-get update.
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Network / security
    ca-certificates \
    curl \
    libnss3 \
    libnspr4 \
    # Accessibility / AT-SPI bridge
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    # Printing
    libcups2 \
    # DRM / GPU
    libdrm2 \
    libgbm1 \
    # Input / display
    libxkbcommon0 \
    libx11-6 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libxshmfence1 \
    # Audio
    libasound2 \
    # Fonts
    fonts-liberation \
    fonts-noto-color-emoji \
    # Process helpers
    procps \
    && playwright install chromium --with-deps \
    && rm -rf /var/lib/apt/lists/*

# ── Application code ─────────────────────────────────────────────────────────
WORKDIR /app
COPY src/ ./src/

# ── Runtime environment ──────────────────────────────────────────────────────
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Playwright headless in containers
ENV BROWSER_HEADLESS=true

# ── Non-root user for security ───────────────────────────────────────────────
RUN useradd --no-create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app && \
    chown -R appuser:appuser /ms-playwright
USER appuser

LABEL maintainer="UI Navigator" version="1.4.0"

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
