FROM ubuntu:22.04

LABEL maintainer="Tim Schneider <tim@robot-learning.de>"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y install build-essential cmake git catch2 python3-dev libboost-all-dev liburdfdom-dev libfmt-dev libssl-dev libmysqlclient-dev libpq-dev libtinyxml2-dev

ADD docker/common/ /tmp/common/
ARG LIBFRANKA_VERSION
RUN /tmp/common/install-dependencies

RUN cd /var/tmp  \
    && git clone https://github.com/TimSchneider42/franky.git --recurse-submodules \
    && cd franky \
    && mkdir build && cd build \
    && cmake .. \
    && make -j$(nproc) \
    && make install \
    && cd /var/tmp && rm -rf franky

# Cleaning up
RUN rm -rf /tmp/common
