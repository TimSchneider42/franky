# ⚙️ Installation

To install franky, you have to follow three steps:

1. Ensure that you are using a real-time kernel
2. Ensure that the executing user has permission to run real-time applications
3. Install franky via pip or [build it from source](../development.md)

## Installing a Real-Time Kernel

In order for franky to function properly, it requires the underlying OS to use a real-time kernel.
Otherwise, you might see `communication_constrains_violation` errors.

To check whether your system is currently using a real-time kernel, type `uname -a`.
You should see something like this:

```
$ uname -a
Linux [PCNAME] 5.15.0-1056-realtime #63-Ubuntu SMP PREEMPT_RT ...
```

If it does not say PREEMPT_RT, you are not currently running a real-time kernel.

There are multiple ways of installing a real-time kernel.
You can [build it from source](https://frankarobotics.github.io/docs/doc/libfranka/docs/real_time_kernel.html) or, if
you are using Ubuntu, it can be [enabled through Ubuntu Pro](https://ubuntu.com/real-time).

## Allowing the Executing User to Run Real-Time Applications

First, create a group `realtime` and add your user (or whoever is running franky) to this group:

```bash
sudo addgroup realtime
sudo usermod -a -G realtime $(whoami)
```

Afterward, add the following limits to the real-time group in /etc/security/limits.conf:

```
@realtime soft rtprio 99
@realtime soft priority 99
@realtime soft memlock 102400
@realtime hard rtprio 99
@realtime hard priority 99
@realtime hard memlock 102400
```

Log out and log in again to let the changes take effect.

To verify that the changes were applied, check if your user is in the `realtime` group:

```bash
$ groups
... realtime
```

If real-time is not listed in your groups, try rebooting.

## Installing franky

To start using franky with Python and the latest libfranka *0.21.2*, just install it via

```bash
pip install franky-control
```

If you need `franky` for a specific robot server version, you can install the corresponding wheel by adding an `--extra-index-url` specifying your robot's server version. `franky` supports Robot Server versions 2 through 10.

```bash
# Replace 10 with your robot server version
pip install franky-control --extra-index-url "https://timschneider42.github.io/franky/whl/by-robot-server-version/10/"
```

If you need a specific libfranka version, we provide wheels for versions *0.7.1*, *0.8.0*, *0.9.2*, *0.12.1*, *0.13.3*,
*0.14.2*, *0.17.0*, and *0.21.2*.
They can be installed via

```bash
# You can replace the libfranka version by any of the supported versions denoted above
pip install franky-control --extra-index-url "https://timschneider42.github.io/franky/whl/libfranka-0.21.2/"
```

## Development Builds

If you need the latest features before they make it into an official release, we provide wheels of the current `master`
branch in the rolling [dev release](https://github.com/TimSchneider42/franky/releases/tag/dev-release).
These wheels are rebuilt on every push to `master` and are provided for all supported libfranka versions.
They can be installed via the [package index](https://timschneider42.github.io/franky/whl/) by adding the `--pre` flag:

```bash
# Replace 10 with your robot server version
pip install --pre franky-control --extra-index-url "https://timschneider42.github.io/franky/whl/by-robot-server-version/10/"
```

Development builds are versioned as pre-releases of the next patch version, and their version indicates the commit and
libfranka version they were built against: e.g., if the latest release is *1.1.4*, then
*1.1.5.dev1234+g8cb09e5.libfranka.0.9.2* is a development build of commit `8cb09e5` on `master` for libfranka *0.9.2*.
