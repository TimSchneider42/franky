# franky — High-Level Control Library for Franka Robots

franky is a high-level control library for Franka robots, offering Python and C++ support.
By providing a high-level control interface, franky eliminates the need for strict real-time programming at 1 kHz,
making control from non-real-time environments, such as Python programs, feasible.
Instead of relying on low-level control commands, franky expects high-level position or velocity targets and
uses [Ruckig](https://github.com/pantor/ruckig) to plan time-optimal trajectories in real-time.

Although Python does not provide real-time guarantees, franky strives to maintain as much real-time control as possible.
Motions can be preempted at any moment, prompting franky to re-plan trajectories on the fly.
To handle unforeseen situations — such as unexpected contact with the environment — franky includes a reaction system
that allows for updating motion commands dynamically.
Furthermore, most non-real-time functionality of
[libfranka](https://frankarobotics.github.io/docs/doc/libfranka/docs/index.html), such as gripper control, is made
directly available in Python.

Ready to go? Follow the [installation instructions](getting-started/installation.md) and check out
the [guide](guide/index.md) for an introduction.
If you do not have a robot at hand, you can also try the [simulation](guide/simulation.md) first.

```{toctree}
:caption: Getting Started
:maxdepth: 1

getting-started/installation
getting-started/docker
getting-started/faq
```

```{toctree}
:caption: Guide
:maxdepth: 2

guide/index
```

```{toctree}
:caption: Development
:maxdepth: 1

development
```
