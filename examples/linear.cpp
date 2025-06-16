#include <franky.hpp>

#include "util.cpp"

using namespace franky;

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <robot-hostname>" << std::endl;
    return -1;
  }

  // Connect to the robot
  Robot robot(argv[1]);
  robot.automaticErrorRecovery();

  // Reduce the acceleration and velocity dynamic
  robot.setRelativeDynamicsFactor(0.1);

  // Define and move forwards
  auto way = mk_affine(0.0, 0.05, 0.0);
  auto motion_forward = std::make_shared<CartesianMotion>(RobotPose(way), ReferenceType::kRelative);
  robot.move(motion_forward);

  // And move backwards using the inverse motion
  auto motion_backward = std::make_shared<CartesianMotion>(RobotPose(way.inverse()), ReferenceType::kRelative);
  robot.move(motion_backward);

  return 0;
}
