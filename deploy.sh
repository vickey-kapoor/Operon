#!/usr/bin/env bash
# =============================================================================
# deploy.sh — Build, push, and deploy UI Navigator to Google Cloud Run
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via environment variables if needed)
# ---------------------------------------------------------------------------
PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${GOOGLE_CLOUD_REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-ui-navigator}"
REPO_NAME="${REPO_NAME:-ui-navigator}"
# Set to "true" to make the service publicly reachable without Cloud Run auth.
# When ALLOW_UNAUTHENTICATED=false the service requires a valid Google ID token;
# use the app-level X-API-Key header for per-client authentication instead.
ALLOW_UNAUTHENTICATED="${ALLOW_UNAUTHENTICATED:-false}"

IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD 2>/dev/null || echo "latest")}"
REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/app"
FULL_IMAGE="${REGISTRY}:${IMAGE_TAG}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info()  { echo "  [INFO]  $*"; }
error() { echo "  [ERROR] $*" >&2; exit 1; }
step()  { echo ""; echo "==> $*"; }

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
step "Pre-flight checks"

command -v gcloud >/dev/null 2>&1 || error "gcloud is not installed. See https://cloud.google.com/sdk/docs/install"
command -v docker  >/dev/null 2>&1 || error "docker is not installed."

# Verify gcloud authentication
if ! gcloud auth print-access-token >/dev/null 2>&1; then
  error "Not authenticated with gcloud. Run: gcloud auth login"
fi

[[ -z "${PROJECT_ID}" ]] && error "Could not determine GCP project ID. Set GOOGLE_CLOUD_PROJECT or run: gcloud config set project YOUR_PROJECT_ID"

info "Project:      ${PROJECT_ID}"
info "Region:       ${REGION}"
info "Service:      ${SERVICE_NAME}"
info "Image:        ${FULL_IMAGE}"

# Enable required APIs
step "Enabling required GCP APIs (if not already enabled)"
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com \
  firestore.googleapis.com \
  monitoring.googleapis.com \
  storage.googleapis.com \
  cloudtrace.googleapis.com \
  --project="${PROJECT_ID}" \
  --quiet

# ---------------------------------------------------------------------------
# Artifact Registry — create repository if it does not exist
# ---------------------------------------------------------------------------
step "Ensuring Artifact Registry repository exists"

if ! gcloud artifacts repositories describe "${REPO_NAME}" \
     --location="${REGION}" \
     --project="${PROJECT_ID}" \
     --quiet >/dev/null 2>&1; then
  info "Creating repository '${REPO_NAME}' in ${REGION}…"
  gcloud artifacts repositories create "${REPO_NAME}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="UI Navigator container images" \
    --project="${PROJECT_ID}" \
    --quiet
  info "Repository created."
else
  info "Repository '${REPO_NAME}' already exists."
fi

# Authenticate Docker to Artifact Registry
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# ---------------------------------------------------------------------------
# Build Docker image
# ---------------------------------------------------------------------------
step "Building Docker image: ${FULL_IMAGE}"

docker build \
  --tag "${FULL_IMAGE}" \
  --tag "${REGISTRY}:latest" \
  --cache-from "${REGISTRY}:latest" \
  --file Dockerfile \
  .

# ---------------------------------------------------------------------------
# Push to Artifact Registry
# ---------------------------------------------------------------------------
step "Pushing image to Artifact Registry"

docker push "${FULL_IMAGE}"
docker push "${REGISTRY}:latest"

# ---------------------------------------------------------------------------
# Deploy to Cloud Run
# ---------------------------------------------------------------------------
step "Deploying to Cloud Run (service: ${SERVICE_NAME})"

# Ensure the GOOGLE_API_KEY secret exists in Secret Manager.
# If the GOOGLE_API_KEY env var is set, store it (create or update).
# Otherwise create a placeholder and prompt the user to update manually.
if [[ -n "${GOOGLE_API_KEY:-}" ]]; then
  info "Storing GOOGLE_API_KEY in Secret Manager…"
  if ! gcloud secrets describe GOOGLE_API_KEY \
       --project="${PROJECT_ID}" \
       --quiet >/dev/null 2>&1; then
    echo -n "${GOOGLE_API_KEY}" | gcloud secrets create GOOGLE_API_KEY \
      --data-file=- \
      --project="${PROJECT_ID}" \
      --quiet
    info "Secret 'GOOGLE_API_KEY' created."
  else
    echo -n "${GOOGLE_API_KEY}" | gcloud secrets versions add GOOGLE_API_KEY \
      --data-file=- \
      --project="${PROJECT_ID}" \
      --quiet
    info "Secret 'GOOGLE_API_KEY' updated with new version."
  fi
else
  if ! gcloud secrets describe GOOGLE_API_KEY \
       --project="${PROJECT_ID}" \
       --quiet >/dev/null 2>&1; then
    info "Secret 'GOOGLE_API_KEY' not found and GOOGLE_API_KEY env var not set. Creating placeholder…"
    echo "REPLACE_ME" | gcloud secrets create GOOGLE_API_KEY \
      --data-file=- \
      --project="${PROJECT_ID}" \
      --quiet
    echo ""
    echo "  *** ACTION REQUIRED ***"
    echo "  Update the secret value before the service can call Gemini:"
    echo "  echo -n 'YOUR_REAL_API_KEY' | gcloud secrets versions add GOOGLE_API_KEY --data-file=- --project=${PROJECT_ID}"
    echo ""
  fi
fi

# ---------------------------------------------------------------------------
# GCS bucket for screenshot storage
# ---------------------------------------------------------------------------
step "Ensuring GCS screenshot bucket exists"

GCS_BUCKET="${GCS_BUCKET:-${PROJECT_ID}-ui-navigator-screenshots}"

if ! gcloud storage buckets describe "gs://${GCS_BUCKET}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  info "Creating bucket gs://${GCS_BUCKET} in ${REGION}…"
  gcloud storage buckets create "gs://${GCS_BUCKET}" --project="${PROJECT_ID}" --location="${REGION}" --quiet
  info "Bucket created."
else
  info "Bucket gs://${GCS_BUCKET} already exists."
fi

# ---------------------------------------------------------------------------
# Cloud Run service account + IAM bindings
# ---------------------------------------------------------------------------
step "Configuring Cloud Run service account and IAM bindings"

SA_NAME="${SERVICE_NAME}-sa"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

if ! gcloud iam service-accounts describe "${SA_EMAIL}" \
     --project="${PROJECT_ID}" \
     --quiet >/dev/null 2>&1; then
  info "Creating service account ${SA_EMAIL}…"
  gcloud iam service-accounts create "${SA_NAME}" \
    --display-name="UI Navigator Cloud Run SA" \
    --project="${PROJECT_ID}" \
    --quiet
fi

# Grant required roles
for ROLE in roles/datastore.user roles/monitoring.metricWriter roles/cloudtrace.agent; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="${ROLE}" \
    --condition=None \
    --quiet >/dev/null
done

# Grant storage.objectAdmin on the screenshots bucket only
gcloud storage buckets add-iam-policy-binding "gs://${GCS_BUCKET}" --member="serviceAccount:${SA_EMAIL}" --role="roles/storage.objectAdmin" --quiet >/dev/null

info "IAM bindings configured."

# Build the auth flag based on ALLOW_UNAUTHENTICATED setting.
if [[ "${ALLOW_UNAUTHENTICATED}" == "true" ]]; then
  AUTH_FLAG="--allow-unauthenticated"
  info "Cloud Run auth: public (unauthenticated access allowed)"
else
  AUTH_FLAG="--no-allow-unauthenticated"
  info "Cloud Run auth: private (requires Cloud Run invoker role or app-level API key)"
fi

gcloud run deploy "${SERVICE_NAME}" \
  --image="${FULL_IMAGE}" \
  --region="${REGION}" \
  --platform=managed \
  ${AUTH_FLAG} \
  --service-account="${SA_EMAIL}" \
  --memory=2Gi \
  --cpu=2 \
  --concurrency=10 \
  --min-instances=0 \
  --max-instances=5 \
  --timeout=300 \
  --set-secrets="GOOGLE_API_KEY=GOOGLE_API_KEY:latest" \
  --set-env-vars="BROWSER_HEADLESS=true,LOG_LEVEL=INFO,MAX_CONCURRENT_TASKS=5,BROWSER_WIDTH=1280,BROWSER_HEIGHT=800,RATE_LIMIT_RPM=60,TASK_STORE=firestore,GCS_BUCKET=${GCS_BUCKET}" \
  --project="${PROJECT_ID}" \
  --quiet

# ---------------------------------------------------------------------------
# Print service URL
# ---------------------------------------------------------------------------
step "Deployment complete"

SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --format="value(status.url)")

echo ""
echo "  ✅  UI Navigator is live at:"
echo "      ${SERVICE_URL}"
echo ""
echo "  Health check:  ${SERVICE_URL}/health"
echo "  API docs:      ${SERVICE_URL}/docs"
echo ""
echo "  Example usage:"
echo "    curl -X POST ${SERVICE_URL}/navigate \\"
echo "         -H 'Content-Type: application/json' \\"
echo "         -H 'X-API-Key: YOUR_API_KEY' \\"
echo "         -d '{\"task\": \"Go to example.com and report the page title\", \"max_steps\": 5}'"
echo ""
