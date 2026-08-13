# 🐋 Using Docker

To use franky within Docker we provide
a [Dockerfile](https://github.com/TimSchneider42/franky/blob/master/docker/run/Dockerfile) and
accompanying [docker-compose](https://github.com/TimSchneider42/franky/blob/master/docker-compose.yml) file.

```bash
git clone --recurse-submodules https://github.com/timschneider42/franky.git
cd franky/
docker compose build franky-run
```

To use another version of libfranka than the default (0.21.2), add a build argument:

```bash
docker compose build franky-run --build-arg LIBFRANKA_VERSION=0.9.2
```

To run the container:

```bash
docker compose run franky-run bash
```

The container requires access to the host machine's network *and* elevated user rights to allow the Docker user to set
RT capabilities of the processes run from within it.

For building franky and its wheels inside Docker, see [Development](../development.md).
