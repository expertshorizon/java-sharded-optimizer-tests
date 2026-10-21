# Canonical Java base image (Dockerfile & Image Best Practices, "Java (JDK)"
# row), pulled from the sanctioned ECR mirror rather than Docker Hub.
FROM public.ecr.aws/docker/library/eclipse-temurin:21-jdk-jammy@sha256:25d1276565738d3c805e632a4542c3a7598866ef967f4def6544c15de3a74b14

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        asciinema \
        ca-certificates \
        tmux \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY app/ /app/

ARG IMAGE_SOURCE="terminus-3/tasks/java-sharded-optimizer"
ARG IMAGE_REVISION="unknown"
ARG IMAGE_CREATED="1970-01-01T00:00:00Z"
ARG IMAGE_VERSION="v1"
LABEL org.opencontainers.image.title="java-sharded-optimizer" \
      org.opencontainers.image.description="Terminal-Bench task environment: Java sharded optimizer simulator with rational arithmetic" \
      org.opencontainers.image.source="${IMAGE_SOURCE}" \
      org.opencontainers.image.revision="${IMAGE_REVISION}" \
      org.opencontainers.image.created="${IMAGE_CREATED}" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.licenses="MIT"
