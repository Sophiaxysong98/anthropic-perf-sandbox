set -e

docker build --build-arg SANDBOX_MODE=standard \
  -t gcr.io/gemini-frontier-evals/anthropic-perf-sandbox-standard:latest .
docker push gcr.io/gemini-frontier-evals/anthropic-perf-sandbox-standard:latest

docker build --build-arg SANDBOX_MODE=easy \
  -t gcr.io/gemini-frontier-evals/anthropic-perf-sandbox-easy:latest .
docker push gcr.io/gemini-frontier-evals/anthropic-perf-sandbox-easy:latest
