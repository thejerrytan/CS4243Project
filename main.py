import numpy as np
import math
import numpy.linalg as la
import cv2
from homography import *
from corner import *

# We define world origin as (0,0,0) to be at the center of the volleyball court.
# The volleyball court floor is used as the plane for homography calculation purposes
# Hence, because of our choice of coordinates, the world coordinates coincide with our plane coordinates
# with (Wx, Wy, Wz) = (Vp, Up, Zp)

# Video Dimensions - 300 x 631
CLIP1 = './beachVolleyball/beachVolleyball1.mov'
CLIP2 = './beachVolleyball/beachVolleyball2.mov'
CLIP3 = './beachVolleyball/beachVolleyball3.mov'
CLIP4 = './beachVolleyball/beachVolleyball4.mov'
CLIP5 = './beachVolleyball/beachVolleyball5.mov'
CLIP6 = './beachVolleyball/beachVolleyball6.mov'
CLIP7 = './beachVolleyball/beachVolleyball7.mov'

# Using 1 cm = 1 unit as our scale, we can write down the coordinates of 5 key points on the plane
# For purposes of feature extraction, we need to identify points that are good corners and appear consistently
# in all frames of the clip
# Size of olympic sized beach volleyball court - 8m wide by 16m long
VCOURT_CENTER    = np.array([0,0,0])
VCOURT_TOP_LEFT  = np.array([-4, -8, 0])
VCOURT_TOP_RIGHT = np.array([4, -8, 0])
VCOURT_BOT_LEFT  = np.array([-4, 8, 0])
VCOURT_BOT_RIGHT = np.array([4, 8, 0])
VCOURT_NET_LEFT  = np.array([-5,0,0])
VCOURT_NET_RIGHT = np.array([5,0,0])

def get_plane_coordinates(H, img):
	""" Given img point, [uc, vc, 1], apply Homography to obtain the plane/world coordinates of the players (up, up, 1)"""
	return np.dot(np.inv(H), img)

def get_bg(filename):
	""" Get background of image by averaging method. Only works for stationary camera and background"""
	cap = cv2.VideoCapture(filename)
	fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = int(cap.get(cv2.CAP_PROP_FPS))
	fc  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	ret, frame = cap.read()
	count  = 1
	avgImg = np.zeros(frame.shape, dtype=np.float32)
	while(cap.isOpened() and ret):
		alpha = 1.0 / count
		cv2.accumulateWeighted(frame, avgImg, alpha)
		print("Frame = " + str(count) + ", alpha = " + str(alpha))
		cv2.imshow('background', avgImg)
		cv2.waitKey(25)
		normImg = cv2.convertScaleAbs(avgImg)
		cv2.waitKey(25)
		cv2.imshow('normalized background', normImg)
		ret, frame = cap.read()
		count += 1
	print("Frame width       : %d" % fw)
	print("Frame height      : %d" % fh)
	print("Frames per second : %d" % fps)
	print("Frame count       : %d" % fc)
	cv2.imwrite('bg.jpg', normImg)
	cap.release()
	cv2.destroyAllWindows()

def show_video_in_matplot(image):
	plt.figure()
	plt.imshow(image)
	plt.show()

def main():
	cap = cv2.VideoCapture(CLIP1)
	ret, frame = cap.read()
	while cap.isOpened() and ret:
		bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# (corner, x, y) = harris_corner(bw)
		cv2.imshow('background', frame)
		cv2.waitKey(25)
		ret, frame = cap.read()
	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()
