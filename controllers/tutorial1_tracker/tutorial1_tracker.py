"""Tutorial 1: : The goal of this tutorial is to get familiar with the environment and program a first example of
HRI with the humanoid robot Nao"""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Keyboard, Display, Motion, DistanceSensor
import numpy as np
import cv2


class MyRobot(Robot):
    def __init__(self, ext_camera_flag):
        super(MyRobot, self).__init__()
        print('> Starting robot controller')

        self.timeStep = 32  # Milisecs to process the data (loop frequency) - Use int(self.getBasicTimeStep()) for
        # default
        self.state = 0  # Idle starts for selecting different states

        # Sensors init
        self.gps = self.getGPS('gps')
        self.gps.enable(self.timeStep)

        self.step(self.timeStep)  # Execute one step to get the initial position
        self.cameraBottom = self.getCamera("CameraBottom")
        self.cameraBottom.enable(2 * self.timeStep)

        self.ext_camera = ext_camera_flag
        self.displayCamExt = self.getDisplay('CameraExt')

        # external camera
        if self.ext_camera:
            self.cameraExt = cv2.VideoCapture(0)

        # Actuators init
        self.head_yaw = self.getMotor('HeadYaw')
        self.head_yaw.setPosition(float('inf'))
        self.head_yaw.setVelocity(0)

        self.head_pitch = self.getMotor('HeadPitch')
        self.head_pitch.setPosition(float('inf'))
        self.head_pitch.setVelocity(0)

        self.rights_pitch = self.getMotor('RShoulderPitch')
        self.rights_pitch.setPosition(float('inf'))
        self.rights_pitch.setVelocity(0)

        self.rightS_roll = self.getMotor('RShoulderRoll')
        self.rightS_roll.setPosition(float('inf'))
        self.rightS_roll.setVelocity(0)

        # Face Movement
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        # Keyboard
        self.keyboard.enable(self.timeStep)
        self.keyboard = self.getKeyboard()

        self.motion = None
        self.loadMotionFiles()

    def loadMotionFiles(self):
        """
            Loading the pre-defined motions for the robot
        """

        self.handWave = Motion('../../motions/HandWave.motion')
        self.forwards = Motion('../../motions/Forwards50.motion')
        self.backwards = Motion('../../motions/Backwards.motion')
        self.sideStepLeft = Motion('../../motions/SideStepLeft.motion')
        self.sideStepRight = Motion('../../motions/SideStepRight.motion')
        self.turnLeft60 = Motion('../../motions/TurnLeft60.motion')
        self.turnRight60 = Motion('../../motions/TurnRight60.motion')
        self.shoot = Motion('../../motions/Shoot.motion')

    # Captures the external camera frames
    # Returns the image downsampled by 2
    def camera_read_external(self):
        """
            Use the external camera and captures the image.
        :return: img
        :rtype: numpy.ndarray()
        """
        img = []
        if self.ext_camera:
            # Capture frame-by-frame
            ret, frame = self.cameraExt.read()
            # Our operations on the frame come here
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # From openCV BGR to RGB
            img = cv2.resize(img, None, fx=0.5, fy=0.5)  # image downsampled by 2
            print(type(img))
        return img

    # Displays the image on the webots camera display interface
    def image_to_display(self, img):
        if self.ext_camera:
            height, width, channels = img.shape
            imageRef = self.displayCamExt.imageNew(cv2.transpose(img).tolist(), Display.RGB, width, height)
            self.displayCamExt.imagePaste(imageRef, 0, 0)

    def print_gps(self):
        gps_data = self.gps.getValues()
        print('----------gps----------')
        print(' [x y z] =  [' + str(gps_data[0]) + ',' + str(gps_data[1]) + ',' + str(gps_data[2]) + ']')

    @staticmethod
    def printHelp():
        print(
            'Commands:\n'
            ' H for displaying the commands\n'
            ' G for print the gps\n'
            ' Up Arrow to move forward\n'
            ' Down Arrow to move backward\n'
            ' Left Arrow to move head left\n'
            ' Right Arrow to move head left\n'
            ' S to Stop\n'
            ' E to exit\n'
        )

    def startMotion(self, motion):
        """
            Starts the motion and if any previous motion is on stops it before starting a new motion
        :param motion: Class variable
        :type motion: Motion
        """
        if self.motion:
            self.motion.stop()
        motion.play()
        self.motion = motion

    def run_keyboard(self):
        """
            Takes the input from the keyboard and runs the required module
        """
        self.printHelp()

        # Main loop.
        while True:
            # Deal with the pressed keyboard key.
            k = self.keyboard.getKey()
            if k == ord('G'):
                self.print_gps()
            elif k == ord('H'):
                self.printHelp()

            elif k == Keyboard.UP:
                self.startMotion(self.forwards)
            elif k == Keyboard.DOWN:
                self.startMotion(self.backwards)
            elif k == Keyboard.LEFT:
                self.head_yaw.setPosition(float('inf'))
                self.head_yaw.setVelocity(-1)
            elif k == Keyboard.RIGHT:
                self.head_yaw.setPosition(float('inf'))
                self.head_yaw.setVelocity(1.0)
            elif k == ord('L'):
                self.startMotion(self.turnLeft60)
            elif k == ord('R'):
                self.startMotion(self.turnRight60)
            elif k == ord('S'):
                if self.motion:
                    self.motion.stop()
                self.head_yaw.setVelocity(0)

            elif k == ord("E"):
                return

            # Perform a simulation step, quit the loop when
            # Webots is about to quit.
            if self.step(self.timeStep) == -1:
                break

        # finallize class. Destroy external camera.
        if self.ext_camera:
            self.cameraExt.release()

            # Face following main function

    def look_at(self, x, y):
        """
            Translate the point so that the bot follows the face. Takes 2 co-ordinates and positions the bot head
            so it mimics the moment of the face.
            It sets the head movement position by accessing the motor through the head_yaw and head_pitch class
            variable

        :param x: x co-ordinate
        :type x: float()
        :param y: y co-ordinate
        :type y: float()
        """
        x_mov = float((x / 270) - 0.5) * 2
        y_mov = float((y / 270) - 0.5) * 2
        if x_mov > 2.09:
            x_mov = 2.09
        elif x_mov < -2.09:
            x_mov = -2.09

        if y_mov > 0.51:
            y_mov = 0.51
        elif y_mov < -0.67:
            y_mov = -0.67

        # self.motion = robot.getMotor('HeadYaw')
        self.head_yaw.setVelocity(1)
        self.head_yaw.setPosition(x_mov)

        # self.motion = robot.getMotor('HeadPitch')
        self.head_pitch.setVelocity(1)
        self.head_pitch.setPosition(y_mov)

    def run_face_follower(self):
        """
            Using the Haar Cascades classifier detects the face through the external camera and translate the bot head
            movement based on the position of the face in the frame.
        """

        # main control loop: perform simulation steps of self.timeStep milliseconds
        # and leave the loop when the simulation is over
        while self.step(self.timeStep) != -1:

            img = self.camera_read_external()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                w_half = w // 2
                h_half = h // 2
                cv2.circle(img, (x + h_half, y + w_half), 2, (0, 255, 0), -1)
                self.look_at(x + h_half, y + w_half)
            self.image_to_display(img)

        # finalize class. Destroy external camera.
        if self.ext_camera:
            self.cameraExt.release()

        # create the Robot instance and run the controller

    def run_greetings(self):
        """
            Plays a random movement if the bot detects a face or a ball in the internal or external camera.
        """

        while self.step(self.timeStep) != -1:

            img = self.camera_read_external()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            self.image_to_display(img)
            for (x, y, w, h) in faces:
                self.startMotion(self.shoot)

    def detect_ball(self):
        """
            Detects the ball through the internal camera and sets the position of the motors in the head to track the
            movement of the ball. The motors are set to position control and runs with an velocity of 1.
        """

        img = self.cameraBottom.getImage()
        height = self.cameraBottom.getHeight()
        width = self.cameraBottom.getWidth()
        # turn into np array
        image = np.frombuffer(img, np.uint8).reshape(height, width, 4)
        # transform into HSV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # green is defined as tripels between these values
        lower = np.array([40, 100, 100])
        upper = np.array([80, 255, 255])
        # lay a mask over all green values
        mask = cv2.inRange(image, lower, upper)
        # Image preparation
        # Erosion
        kernel = np.ones((2, 2), np.uint8)
        image = cv2.erode(mask, kernel, iterations=1)
        # Dilation
        image = cv2.dilate(image, kernel, iterations=1)
        # find ball contour
        contour, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contour) != 0:
            contour = np.asarray(contour[0])
            m = cv2.moments(contour)
            # if area of detected blob is reasonable:
            if 500 > m["m00"] > 1:
                # calculate momentum
                cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
                return cx, cy
            else:
                return None, None

        yaw_position = 0
        pitch_position = 0
        self.head_yaw.setPosition(float('inf'))
        self.head_pitch.setPosition(float('inf'))
        self.head_yaw.setVelocity(1)
        self.head_pitch.setVelocity(1)

        while self.step(self.timeStep) != -1:

            x, y = self.detect_ball()

            if x is None:
                continue
            else:
                K = 0.2
                dx, dy = K * ((x / width) - 0.5), K * ((y / height) - 0.5)
                print(yaw_position - dx, pitch_position + dy)
                yaw_position = yaw_position - dx
                pitch_position = pitch_position + dy

                if yaw_position > 1.8:
                    yaw_position = 1.8

                elif yaw_position < -1.8:
                    yaw_position = -1.8

                if pitch_position > 0.5:
                    pitch_position = 0.5

                elif pitch_position < -0.5:
                    pitch_position = -0.5

                self.head_yaw.setPosition(float(yaw_position))
                self.head_pitch.setPosition(float(pitch_position))

        return None, None


robot = MyRobot(ext_camera_flag=True)
# robot.run_keyboard()
# robot.run_face_follower()
# robot.run_ball_follower()
# robot.detect_ball()
robot.run_greetings()
