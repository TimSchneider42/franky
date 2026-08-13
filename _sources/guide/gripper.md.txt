# 👌 Gripper

In the `franky::Gripper` class, the default gripper force and gripper speed can be set.
Then, in addition to the libfranka commands, the following helper methods can be used:

```c++
#include <franky.hpp>
#include <chrono>
#include <future>

auto gripper = franky::Gripper("10.90.90.1");

double speed = 0.02; // [m/s]
double force = 20.0; // [N]

// Move the fingers to a specific width (5cm)
bool success = gripper.move(0.05, speed);

// Grasp an object of unknown width
success &= gripper.grasp(0.0, speed, force, epsilon_outer=1.0);

// Get the width of the grasped object
double width = gripper.width();

// Release the object
gripper.open(speed);

// There are also asynchronous versions of the methods
std::future<bool> success_future = gripper.moveAsync(0.05, speed);

// Wait for 1s
if (!success_future.wait_for(std::chrono::seconds(1)) == std::future_status::ready) {
  // Get the result
  std::cout << "Success: " << success_future.get() << std::endl;
} else {
  gripper.stop();
  success_future.wait();
  std::cout << "Gripper motion timed out." << std::endl;
}
```

The Python API follows the C++ API closely:

```python
import franky

gripper = franky.Gripper("10.90.90.1")

speed = 0.02  # [m/s]
force = 20.0  # [N]

# Move the fingers to a specific width (5cm)
success = gripper.move(0.05, speed)

# Grasp an object of unknown width
success &= gripper.grasp(0.0, speed, force, epsilon_outer=1.0)

# Get the width of the grasped object
width = gripper.width

# Release the object
gripper.open(speed)

# There are also asynchronous versions of the methods
success_future = gripper.move_async(0.05, speed)

# Wait for 1s
if success_future.wait(1):
    print(f"Success: {success_future.get()}")
else:
    gripper.stop()
    success_future.wait()
    print("Gripper motion timed out.")
```
