FROM quay.io/pypa/manylinux_2_34_x86_64:latest

LABEL maintainer="Tim Schneider <tim@robot-learning.de>"

RUN yum -y install poco-devel* zip boost1.78-devel urdfdom-devel fmt-devel gcc-g++

RUN ln -s /usr/lib64/cmake/Boost-1.78.0/ /usr/lib64/cmake/Boost
ENV LD_LIBRARY_PATH="/usr/local/lib64"

ADD docker/common/ /tmp/common/
ARG LIBFRANKA_VERSION
RUN /tmp/common/install-dependencies

ARG PYBIN_SUPPORTED_VERSIONS="cp3(7m|8|9|10|11|12|13)"
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
ENV PYBIN_SUPPORTED_VERSIONS=${PYBIN_SUPPORTED_VERSIONS}
ENV LIBFRANKA_VERSION=${LIBFRANKA_VERSION}
