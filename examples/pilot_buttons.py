import argparse

from franky import RobotWebSession


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Listen to Franka Desk Pilot button events."
    )
    parser.add_argument("host", type=str, help="FCI IP of the robot")
    parser.add_argument("user", type=str, help="Login username of Franka Desk")
    parser.add_argument("password", type=str, help="Login password of Franka Desk")

    args = parser.parse_args()

    with RobotWebSession(args.host, args.user, args.password) as session:
        print("Listening for Pilot button events. Press Ctrl+C to stop.")
        try:
            while True:
                for event in session.poll_buttons(timeout=1.0):
                    print(event)
        except KeyboardInterrupt:
            pass
