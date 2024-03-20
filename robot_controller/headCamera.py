import sys, rospy, cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class getHeadCamera:
    def __init__(self):
        ## Sends a message of a message type to the selected ROS topic ##
        self.image_pub = rospy.Publisher("/coordinates_from_opencv_head",Image, queue_size=10)
        ## Calls the function CvBridge for python module cv_bridge ##
        self.bridge = CvBridge()
        ## Connect to a ros topic to be able to receieve the data & pass it into the callback function ##
        self.image_sub = rospy.Subscriber("/cameras/head_camera/image",Image,self.callback)
        ## Initalises empty image ##
        self.cv_image = []

    def callback(self,data):
        try:
            ## Gets the ROS image message & converts it to OpenCv & stores it in cv_image array ##
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    ## Returns the converted image ##
    def getCVimage(self):
        return self.cv_image
        
## --MAIN PROGRAM-- ##
def main():
    ## Intilises the head camera class ##
    getCam = getHeadCamera()
    rospy.init_node('getHeadCamera', anonymous=True)
    ## Infinite loop to keep getting the camera image ##
    while (True):
        ## Retrieves the stored converted image ##
        img = getCam.getCVimage()

        ## Display the ROS topic being used ##
        cv2.imshow("Head Camera", img)

        ## Displays the image in OpenCV ##
        key = cv2.waitKey(1) & 0xFF
        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break

if __name__ == "__main__":
    main()