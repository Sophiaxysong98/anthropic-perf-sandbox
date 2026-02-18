FROM python:3.11-slim

ARG SANDBOX_MODE=standard

RUN apt-get update && apt-get install -y \
    git \
    curl \
    gosu \
    e2fsprogs \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash agent

ARG GITHUB_REPO=Sophiaxysong98/anthropic-perf-sandbox
RUN git clone https://github.com/${GITHUB_REPO}.git /opt/perf_takehome

RUN if [ "$SANDBOX_MODE" = "easy" ]; then \
      cp /opt/perf_takehome/sandbox_easy/perf_takehome_preoptimized.py /opt/perf_takehome/perf_takehome.py; \
    fi

RUN mkdir -p /opt/perf_takehome/tests && \
    cp /opt/perf_takehome/sandbox_${SANDBOX_MODE}/frozen_problem.py /opt/perf_takehome/tests/frozen_problem.py && \
    cp /opt/perf_takehome/sandbox_${SANDBOX_MODE}/submission_tests.py /opt/perf_takehome/tests/submission_tests.py

RUN chmod -R 555 /opt/perf_takehome/tests && \
    chmod 444 /opt/perf_takehome/tests/*.py && \
    chown -R root:root /opt/perf_takehome/tests

RUN mkdir -p /home/agent/perf_takehome && \
    chown -R agent:agent /home/agent

RUN cp -r /opt/perf_takehome/tests /home/agent/perf_takehome/tests && \
    chown -R root:root /home/agent/perf_takehome/tests && \
    chmod -R 555 /home/agent/perf_takehome/tests

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /home/agent/perf_takehome

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

CMD ["sleep", "infinity"]
