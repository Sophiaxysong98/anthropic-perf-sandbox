#!/bin/bash
# Load sandbox images before running the eval.
# Pulls from GCR instead of loading embedded tarballs.

PROJECT_ID="${PROJECT_ID:-gemini-frontier-evals}"

echo "Pulling sandbox images from GCR..."
docker pull "gcr.io/${PROJECT_ID}/anthropic-perf-sandbox-standard:latest" 2>/dev/null && echo "Pulled sandbox_standard" || echo "WARN: Failed to pull sandbox_standard"
docker pull "gcr.io/${PROJECT_ID}/anthropic-perf-sandbox-easy:latest" 2>/dev/null && echo "Pulled sandbox_easy" || echo "WARN: Failed to pull sandbox_easy"

# Also try loading from embedded tarballs as fallback
if [[ -f /app/sandbox_images/sandbox_standard.tar ]]; then
    docker load < /app/sandbox_images/sandbox_standard.tar 2>/dev/null && echo "Loaded sandbox_standard from tarball"
fi
if [[ -f /app/sandbox_images/sandbox_easy.tar ]]; then
    docker load < /app/sandbox_images/sandbox_easy.tar 2>/dev/null && echo "Loaded sandbox_easy from tarball"
fi

echo "Available Docker images:"
docker images | grep -E "sandbox|REPOSITORY"

# Execute the original command
exec "$@"
