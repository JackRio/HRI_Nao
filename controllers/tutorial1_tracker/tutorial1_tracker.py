"""tutorial1_tracker controller."""

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
        self.cameraBottom.enable(2*self.timeStep)

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

       # self.motion = None

        # Face Movement
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        # Keyboard
        self.keyboard.enable(self.timeStep)
        self.keyboard = self.getKeyboard()

        self.currentlyPlaying = False
        self.loadMotionFiles()

    def loadMotionFiles(self):
        self.handWave = Motion('../../motions/HandWave.motion')
        self.forwards = Motion('../../motions/Forwards50.motion')
        self.backwards = Motion('../../motions/Backwards.motion')
        self.sideStepLeft = Motion('../../motions/SideStepLeft.motion')
        self.sideStepRight = Motion('../../motions/SideStepRight.motion')
        self.turnLeft60 = Motion('../../motions/TurnLeft60.motion')
        self.turnRight60 = Motion('../../motions/TurnRight60.motion')

    def startMotion(self, motion):
        # interrupt current motion
        if self.currentlyPlaying:
            self.currentlyPlaying.stop()

        # start new motion
        motion.play()
        self.currentlyPlaying = motion


    # Captures the external camera frames
    # Returns the image downsampled by 2
    def camera_read_external(self):
        img = []
        if self.ext_camera:
            # Capture frame-by-frame
            ret, frame = self.cameraExt.read()
            # Our operations on the frame come here
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # From openCV BGR to RGB
            img = cv2.resize(img, None, fx=0.5, fy=0.5)  # image downsampled by 2

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

    def printHelp(self):
        print(
            'Commands:\n'
            ' H for displaying the commands\n'
            ' G for print the gps\n'
            ' Up Arrow to move forward\n'
            ' Down Arrow to move backward\n'
            ' Left Arrow to move head left\n'
            ' Right Arrow to move head left\n'
            ' S to Stop\n'
            'E to exit\n'
        )

    def run_keyboard(self):

        self.printHelp()
        previous_message = ''

        # Main loop.
        while True:
            # Deal with the pressed keyboard key.
            k = self.keyboard.getKey()
            message = ''
            if k == ord('G'):
                self.print_gps()
            elif k == ord('H'):
                self.printHelp()

            elif k == Keyboard.UP:
                self.startMotion(self.forwards)
            elif k == Keyboard.DOWN:
                self.startMotion(self.backwards)
            elif k == Keyboard.LEFT:
                self.head_yaw.setVelocity(-1)
            elif k == Keyboard.RIGHT:
                self.head_yaw.setVelocity(1.0)
            elif k == ord('L'):
                self.startMotion(self.turnLeft60)
            elif k == ord('R'):
                self.startMotion(self.turnRight60)
            elif k == ord('S'):
                if self.currentlyPlaying:
                    self.currentlyPlaying.stop()
                    self.currentlyPlaying = False

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
        x_mov = float((x / 270) - 0.5) * 4
        y_mov = float((y / 270) - 0.5) * 4
        if x_mov > 2.09:
            x_mov = 2.09
        elif x_mov < -2.09:
            x_mov = -2.09

        if y_mov > 0.51:
            y_mov = 0.51
        elif y_mov < -0.67:
            y_mov = -0.67

        self.motion = robot.getMotor('HeadYaw')
        self.motion.setPosition(x_mov)

        self.motion = robot.getMotor('HeadPitch')
        self.motion.setPosition(y_mov)

    def run_face_follower(self):
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

        # finallize class. Destroy external camera.
        if self.ext_camera:
            self.cameraExt.release()

        # create the Robot instance and run the controller

    def run_ball_follower(self):

        yaw_position = 0
        pitch_position = 0
        self.head_yaw.setVelocity(6.5)
        self.head_pitch.setVelocity(6.5)


        while self.step(self.timeStep) != -1:
            img = self.cameraBottom.getImage()
            height, width = self.cameraBottom.getHeight(), self.cameraBottom.getWidth()
            image = np.frombuffer(img, np.uint8).reshape( height, width,4)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower =  np.array([40,100,100])
            upper = np.array([80,255,255])
            # mask green values
            mask = cv2.inRange(image, lower, upper)

            #Erosion
            kernel = np.ones((2,2),np.uint8)
            image = cv2.erode(mask,kernel,iterations = 1)
            #Dilation
            image= cv2.dilate(image,kernel,iterations = 1)
            # find ball contour
            contour, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
            if len(contour) != 0:
                contour = np.asarray(contour[0])
                m = cv2.moments(contour)
                # if area of detected blob is reasonable:
                if 500 > m["m00"] > 1:
                    # calculate momentum
                    cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
                    # calculate movemnet for yaw and pitch
                    K = 0.2
                    dx, dy =K*((cx/width)-0.5), K*((cy/height)-0.5)
                    # allowed values between -2 and 2
                    # Sensible values between -1.5 and 1.5
                    if -1.5 < yaw_position - dx < 1.5:
                        yaw_position = yaw_position - dx
                        self.head_yaw.setPosition(float(yaw_position))
                    # allowed values between -0,6 and 0.5
                    # Sensible values between -0.4 and 0.3
                    if -0.4 < pitch_position + dy < 0.3:
                        pitch_position = pitch_position + dy
                        self.head_pitch.setPosition(float((pitch_position)))
                        #self.rightS_pitch.setPosition(float(
                        #self.startMotion(self.handWave)


robot = MyRobot(ext_camera_flag=False)
robot.run_keyboard()
robot.run_face_follower()
robot.run_ball_follower()
