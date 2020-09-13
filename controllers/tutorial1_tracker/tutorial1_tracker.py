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

        self.ext_camera = ext_camera_flag
        self.displayCamExt = self.getDisplay('CameraExt')

        # external camera
        if self.ext_camera:
            self.cameraExt = cv2.VideoCapture(0)

        # Actuators init
        self.walk_forward = Motion('../../motions/Forwards.motion')
        self.walk_backward = Motion('../../motions/Backwards.motion')

        self.motion = None
        
        #Face Movement
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        # Keyboard
        self.keyboard.enable(self.timeStep)
        self.keyboard = self.getKeyboard()

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
        )

    def move_forward(self):

        self.walk_forward.play()
        self.walk_forward.setLoop(True)

    def move_backward(self):

        self.walk_backward.play()
        self.walk_backward.setLoop(True)

    def stop(self):
        self.walk_forward.stop()
        self.walk_backward.stop()
        if self.motion:
            self.motion.setVelocity(0)

    def head_left(self):
        self.motion = robot.getMotor('HeadYaw')
        self.motion.setPosition(float('inf'))
        self.motion.setVelocity(-1.0)

    def head_right(self):
        self.motion = robot.getMotor('HeadYaw')
        self.motion.setPosition(float('inf'))
        self.motion.setVelocity(1.0)

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
            elif k == 315:
                self.move_forward()
            elif k == 317:
                self.move_backward()
            elif k == 314:
                self.head_left()
            elif k == 316:
                self.head_right()
            elif k == ord('S'):
                self.stop()
            elif k == ord('E'):
                return

            # Perform a simulation step, quit the loop when
            # Webots is about to quit.
            if self.step(self.timeStep) == -1:
                break

        # finallize class. Destroy external camera.
        if self.ext_camera:
            self.cameraExt.release()

            # Face following main function

    def look_at(self):
        pass

    def run_face_follower(self):
        # main control loop: perform simulation steps of self.timeStep milliseconds
        # and leave the loop when the simulation is over
        while self.step(self.timeStep) != -1:
            
            img = self.camera_read_external()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                w_half = w//2
                h_half = h//2
                cv2.circle(img , (x + h_half,y + w_half), 2, (0, 255, 0), -1)
                
            self.image_to_display(img)
        # finallize class. Destroy external camera.
        if self.ext_camera:
            self.cameraExt.release()

        # create the Robot instance and run the controller


robot = MyRobot(ext_camera_flag=True)
robot.run_keyboard()
robot.run_face_follower()
# robot.run_ball_follower()