ARG MANYLINUX_VERSION=manylinux_2_34_x86_64
FROM quay.io/pypa/${MANYLINUX_VERSION}:latest

LABEL maintainer="Tim Schneider <tim@robot-learning.de>"

RUN yum -y install zip fmt-devel openssl-devel mysql-devel postgresql-devel boost-devel

ADD docker/common/ /tmp/common/
ARG LIBFRANKA_VERSION
RUN /tmp/common/install-dependencies
RUN /tmp/common/manylinux-setup-pip

# Set working directory
RUN mkdir -p /code
RUN mkdir -p /build
VOLUME /code
VOLUME /build
WORKDIR /build
ADD docker/common/manylinux-build-wheels /usr/bin/
ADD docker/common/dependency_versions /usr/bin/
ADD docker/build/build-wheels /usr/bin/
ADD docker/build/run-tests /usr/bin/

# Cleaning up
RUN rm -rf /tmp/common

# Preserve arguments for runtime
ENV LIBFRANKA_VERSION=${LIBFRANKA_VERSION}
