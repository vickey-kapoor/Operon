#!/usr/bin/env bash
# =============================================================================
# monitoring/setup_alerts.sh — Create Cloud Monitoring alert policies
# =============================================================================
# Usage:
#   export GOOGLE_CLOUD_PROJECT=my-project
#   export SERVICE_NAME=ui-navigator   # optional, default: ui-navigator
#   chmod +x monitoring/setup_alerts.sh && ./monitoring/setup_alerts.sh
# =============================================================================
set -euo pipefail

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
SERVICE_NAME="${SERVICE_NAME:-ui-navigator}"
NOTIFICATION_CHANNEL="${NOTIFICATION_CHANNEL:-}"  # optional email/PagerDuty channel ID

info()  { echo "  [INFO]  $*"; }
step()  { echo ""; echo "==> $*"; }
error() { echo "  [ERROR] $*" >&2; exit 1; }

[[ -z "${PROJECT_ID}" ]] && error "Set GOOGLE_CLOUD_PROJECT or run: gcloud config set project YOUR_PROJECT_ID"

info "Project: ${PROJECT_ID}"
info "Service: ${SERVICE_NAME}"

# ---------------------------------------------------------------------------
# Helper: create or skip alert policy
# ---------------------------------------------------------------------------
create_policy() {
  local name="$1"
  local display="$2"
  local json="$3"

  info "Creating alert policy: ${display}"
  echo "${json}" | gcloud beta monitoring policies create \
    --policy-from-file=- \
    --project="${PROJECT_ID}" \
    --quiet 2>/dev/null || info "(already exists — skipping)"
}

# ---------------------------------------------------------------------------
# 1. Error rate > 10% (log-based metric: tasks_failed / tasks_started)
# ---------------------------------------------------------------------------
step "Alert 1 — High task failure rate (> 10%)"

create_policy "task-failure-rate" "UI Navigator — High Task Failure Rate" "$(cat <<EOF
{
  "displayName": "UI Navigator — Task Failure Rate > 10%",
  "combiner": "OR",
  "conditions": [
    {
      "displayName": "tasks_failed rate spike",
      "conditionThreshold": {
        "filter": "metric.type=\"custom.googleapis.com/ui_navigator/tasks_failed\" resource.type=\"global\"",
        "aggregations": [
          {
            "alignmentPeriod": "300s",
            "perSeriesAligner": "ALIGN_RATE"
          }
        ],
        "comparison": "COMPARISON_GT",
        "thresholdValue": 0.1,
        "duration": "60s"
      }
    }
  ],
  "alertStrategy": {
    "autoClose": "604800s"
  }
}
EOF
)"

# ---------------------------------------------------------------------------
# 2. p95 request latency > 30s
# ---------------------------------------------------------------------------
step "Alert 2 — High request latency (p95 > 30 s)"

create_policy "request-latency-p95" "UI Navigator — High Request Latency" "$(cat <<EOF
{
  "displayName": "UI Navigator — p95 Request Latency > 30s",
  "combiner": "OR",
  "conditions": [
    {
      "displayName": "request_latency_ms p95 > 30000",
      "conditionThreshold": {
        "filter": "metric.type=\"custom.googleapis.com/ui_navigator/request_latency_ms\" resource.type=\"global\"",
        "aggregations": [
          {
            "alignmentPeriod": "300s",
            "perSeriesAligner": "ALIGN_PERCENTILE_95"
          }
        ],
        "comparison": "COMPARISON_GT",
        "thresholdValue": 30000,
        "duration": "60s"
      }
    }
  ],
  "alertStrategy": {
    "autoClose": "604800s"
  }
}
EOF
)"

# ---------------------------------------------------------------------------
# 3. Cloud Run max instances reached
# ---------------------------------------------------------------------------
step "Alert 3 — Cloud Run max instances reached"

create_policy "cloudrun-max-instances" "UI Navigator — Max Instances Reached" "$(cat <<EOF
{
  "displayName": "UI Navigator — Cloud Run Max Instances Reached",
  "combiner": "OR",
  "conditions": [
    {
      "displayName": "instance_count >= 5",
      "conditionThreshold": {
        "filter": "metric.type=\"run.googleapis.com/container/instance_count\" resource.type=\"cloud_run_revision\" resource.labels.service_name=\"${SERVICE_NAME}\"",
        "aggregations": [
          {
            "alignmentPeriod": "60s",
            "perSeriesAligner": "ALIGN_MAX"
          }
        ],
        "comparison": "COMPARISON_GE",
        "thresholdValue": 5,
        "duration": "60s"
      }
    }
  ],
  "alertStrategy": {
    "autoClose": "604800s"
  }
}
EOF
)"

# ---------------------------------------------------------------------------
step "All alert policies created"
echo ""
echo "  View policies at:"
echo "  https://console.cloud.google.com/monitoring/alerting?project=${PROJECT_ID}"
echo ""
