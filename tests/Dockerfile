# The verifier recompiles the agent's sources with javac, so it needs the same
# JDK 21 the task environment ships. The JDK is copied from the canonical
# Temurin image instead of installing a different JDK major from apt, which
# would reject any Java 18-21 syntax the agent legitimately used.
FROM public.ecr.aws/docker/library/eclipse-temurin:21-jdk-jammy@sha256:25d1276565738d3c805e632a4542c3a7598866ef967f4def6544c15de3a74b14 AS jdk

FROM public.ecr.aws/docker/library/python:3.13-slim-bookworm@sha256:01f42367a0a94ad4bc17111776fd66e3500c1d87c15bbd6055b7371d39c124fb

WORKDIR /tests

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libstdc++6 \
        util-linux \
        zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir --require-hashes --no-deps -r /tmp/requirements.lock \
    && rm -f /tmp/requirements.lock

COPY --from=jdk /opt/java/openjdk /opt/java/openjdk

# The pytest session runs as root and owns the expectations. javac and the
# compiled agent CLI are launched from it as uid 10002 (through setpriv --no-new-privs in
# support.py), which owns nothing but /work: the classes it compiles into and
# the scratch files each invocation reads and writes. /logs/verifier is
# root-owned and 0700, so agent-produced code cannot read or overwrite
# reward.txt, including from a process that outlives the run. /tests is 0750
# for the same reason, which also keeps the golden fixtures and the oracle out
# of reach of the code being graded.
RUN mkdir -p /app /logs/verifier /work \
    && useradd --no-create-home --shell /usr/sbin/nologin --uid 10002 graded \
    && chown root:root /logs/verifier \
    && chmod 0700 /logs/verifier \
    && chown 10002:10002 /work \
    && chmod 0777 /work

ENV JAVA_HOME=/opt/java/openjdk
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Fail the build, not the grading run, if the copied JDK is not usable here.
RUN javac -version && java -version

COPY oracle.py /tests/oracle.py
COPY scenarios.py /tests/scenarios.py
COPY support.py /tests/support.py
COPY test_outputs.py /tests/test_outputs.py
COPY checks_scenarios.py /tests/checks_scenarios.py
COPY checks_checkpoint.py /tests/checks_checkpoint.py
COPY checks_properties.py /tests/checks_properties.py
COPY fixtures/ /tests/fixtures/
COPY --chmod=0755 test.sh /tests/test.sh

RUN chown root:root /tests \
    && chmod 0750 /tests

ARG IMAGE_SOURCE="terminus-3/tasks/java-sharded-optimizer"
ARG IMAGE_REVISION="unknown"
ARG IMAGE_CREATED="1970-01-01T00:00:00Z"
ARG IMAGE_VERSION="v1"
LABEL org.opencontainers.image.title="java-sharded-optimizer-verifier" \
      org.opencontainers.image.description="Terminal-Bench verifier image for java-sharded-optimizer" \
      org.opencontainers.image.source="${IMAGE_SOURCE}" \
      org.opencontainers.image.revision="${IMAGE_REVISION}" \
      org.opencontainers.image.created="${IMAGE_CREATED}" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.licenses="MIT"
