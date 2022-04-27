
from time import sleep


def main(session):
    # pass
    motion_service  = session.service("ALMotion")
    fractionMaxSpeed = 0.2
    while True:
        motion_service.setAngles('LHand', 0.0, fractionMaxSpeed)
        sleep(0.2)
        motion_service.setAngles('LHand', 1.0, fractionMaxSpeed)

        
    

if __name__ == "__main__":

    import argparse
    import qi
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="192.168.1.100",
                        help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")

    args = parser.parse_args()

    try:
        # Initialize qi framework.
        session = qi.Session()
        session.connect("tcp://" + args.ip + ":" + str(args.port))
        print("\nConnected to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) + ".\n")

    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) + ".\n"
                                                                                              "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)
    
    # motion_service  = session.service("ALMotion")
    
    main(session)
    