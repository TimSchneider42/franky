# ❓ FAQ

## I am seeing server version mismatch errors. What now?

If you are seeing errors such as

```
franky.IncompatibleVersionException: libfranka: Incompatible library version (server version: 5, library version: 9)
```

then your Franka robot is either not on the most recent firmware version, or you are using the older Franka Panda
model.
In any case, it's no big deal; just follow our [instructions](installation.md#installing-franky) to install the appropriate franky wheels.

## Can I use CUDA jointly with franky?

Yes. However, you need to set `IGNORE_PREEMPT_RT_PRESENCE=1` during the installation and all subsequent updates of the
CUDA drivers on the real-time kernel.

First, make sure that you have rebooted your system after installing the real-time kernel.
Then, add `IGNORE_PREEMPT_RT_PRESENCE=1` to `/etc/environment`, call `export IGNORE_PREEMPT_RT_PRESENCE=1` to also set
it in the current session, and follow the instructions of Nvidia to install CUDA on your system.

If you are on Ubuntu, you can also
use [this](https://github.com/TimSchneider42/franky/blob/master/tools/install_cuda_realtime.bash) script to install
CUDA on your real-time system:

```bash
# Download the script
wget https://raw.githubusercontent.com/timschneider42/franky/master/tools/install_cuda_realtime.bash

# Inspect the script to ensure it does what you expect

# Make it executable
chmod +x install_cuda_realtime.bash

# Execute the script
./install_cuda_realtime.bash
```

Alternatively, if you are a cowboy and do not care about security, you can also use this one-liner to directly call the
script without checking it:

```bash
bash <(wget -qO- https://raw.githubusercontent.com/timschneider42/franky/master/tools/install_cuda_realtime.bash)
```

## Is your robot connected to a different machine?

No problem!
There are two projects which let you run franky remotely via RPC with minimal
effort: [franky-remote](https://github.com/kvasios/franky-remote) and [net_franky](https://github.com/yblei/net_franky).

Please note that I'm not involved in the development of these projects, so I cannot take any liability for its use.
If you decide to use it, please ensure that you credit the developers of these projects for their work.
