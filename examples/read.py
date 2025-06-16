from argparse import ArgumentParser
from time import sleep

from franky import Affine, Robot


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--host", default="172.16.0.2", help="FCI IP of the robot")
    args = parser.parse_args()

    robot = Robot(args.host)

    while True:
        state = robot.state
        print("\nPose: ", robot.current_pose)
        print("O_TT_E: ", state.O_T_EE)
        print("Joints: ", state.q)
        print("Elbow: ", state.elbow)
        sleep(0.05)
