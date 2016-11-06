from __future__ import division
import numpy as np
import math
import numpy.linalg as la
import cv2
from homography import *
from corner import *
from roi import *
# from imutils.video import VideoStream
import time
np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(threshold=np.inf)
cv2.ocl.setUseOpenCL(False)

# Acknowledgements - The team would like to acknowledge the following resources referenced in our project
# Processing video frames and writing to video file: http://www.pyimagesearch.com/2016/02/22/writing-to-video-with-opencv/
# Lucas-Kanade optical flow tracker -

# We define world origin as (0,0,0) to be at the center of the volleyball court.
# The volleyball court floor is used as the plane for homography calculation purposes
# Hence, because of our choice of coordinates, the world coordinates coincide with our plane coordinates
# with (Wx, Wy, Wz) = (Vp, Up, Zp)
CLIP1 = './beachVolleyball/beachVolleyball1.mov'
CLIP2 = './beachVolleyball/beachVolleyball2.mov'
CLIP3 = './beachVolleyball/beachVolleyball3.mov'
CLIP4 = './beachVolleyball/beachVolleyball4.mov'
CLIP5 = './beachVolleyball/beachVolleyball5.mov'
CLIP6 = './beachVolleyball/beachVolleyball6.mov'
CLIP7 = './beachVolleyball/beachVolleyball7.mov'

# Panoram videos
CLIP1_PAN = './beachVolleyball1_panorama.avi'

# Video Dimensions - 300 x 632
CLIP1_SHAPE = (300, 632)
CLIP2_SHAPE = (300, 632)
CLIP3_SHAPE = (300, 632)
CLIP4_SHAPE = (300, 632)
CLIP5_SHAPE = (300, 632)
CLIP6_SHAPE = (300, 632)
CLIP7_SHAPE = (300, 632)

# Using 1 cm = 1 unit as our scale, we can write down the coordinates of 5 key points on the plane
# For purposes of feature extraction, we need to identify points that are good corners and appear consistently
# in all frames of the clip
# Size of olympic sized beach volleyball court - 8m wide by 16m long
VCOURT_CENTER    = np.array([0,0,0])
VCOURT_TOP_LEFT  = np.array([-400, -800, 0])
VCOURT_TOP_RIGHT = np.array([400, -800, 0])
VCOURT_BOT_LEFT  = np.array([-400, 800, 0])
VCOURT_BOT_RIGHT = np.array([400, 800, 0])
VCOURT_NET_LEFT  = np.array([-500,0,0])
VCOURT_NET_RIGHT = np.array([500,0,0])
VCOURT_LEFT_MID  = np.array([-400,0,0])
VCOURT_RIGHT_MID = np.array([400,0,0])
VCOURT_BOT_MID   = np.array([0,800,0])
VCOURT_TOP_MID   = np.array([0,-800,0])

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
		cv2.waitKey(1)
		normImg = cv2.convertScaleAbs(avgImg)
		cv2.waitKey(1)
		cv2.imshow('normalized background', normImg)
		ret, frame = cap.read()
		count += 1
	print("Frame width       : %d" % fw)
	print("Frame height      : %d" % fh)
	print("Frames per second : %d" % fps)
	print("Frame count       : %d" % fc)
	cv2.imwrite('bg.jpg', normImg)
	cap.release()
	# cv2.destroyAllWindows()
	return normImg

def show_frame_in_matplot(filename, num):
	""" Show frame number from filename in matplot"""
	cap = cv2.VideoCapture(filename)
	count = 0
	while(cap.isOpened() and count <= num):
		count += 1
		ret, frame = cap.read()
	plt.figure()
	plt.imshow(frame)
	plt.show()
	cap.release()

def generate_ROI(shape, x, y, w, h):
	""" Generates region of interest mask, shape is image size (height, width) tuple, x and y are starting coordinates"""
	mask = np.zeros(shape, np.uint8)
	mask[y:y+h, x:x+w] = 255
	return mask

def get_plane_coordinates(H, img):
	""" Given img point, [uc, vc, 1], apply Homography to obtain the plane/world coordinates of the players (up, up, 1)"""
	img = np.matrix(img).T
	try:
		result = np.dot(la.inv(H), img)
		return (result / result[2]).T
	except Exception as e:
		print e
		return np.zeros((3,3))

def plot_player(pts):
	# print(np.max(pts))
	# print(np.min(pts))
	plt.figure()
	plt.ion()
	plt.xlim([-5000,5000])
	plt.ylim([-10000, 10000])
	for i in range(0, pts.shape[0]):
		# if pts[i,2] > -0.8 and pts[i,2] < 1.2:
		plt.scatter(pts[i,0], pts[i,1], marker='x')
		plt.show()
		plt.pause(0.016)
	raw_input()

def constructPanorama(filename, skip=0, end=630):
	# show_frame_in_matplot(CLIP1, 0)
	# ROI_CLIP1_RAND_PT1 = generate_ROI(CLIP1_SHAPE, CLIP1_VCOURT_RAND_PT1['x'], CLIP1_VCOURT_RAND_PT1['y'], CLIP1_VCOURT_RAND_PT1['w'], CLIP1_VCOURT_RAND_PT1['h'])
	# ROI_CLIP1_RAND_PT2 = generate_ROI(CLIP1_SHAPE, CLIP1_VCOURT_RAND_PT2['x'], CLIP1_VCOURT_RAND_PT2['y'], CLIP1_VCOURT_RAND_PT2['w'], CLIP1_VCOURT_RAND_PT2['h'])
	# ROI_CLIP1_RAND_PT3 = generate_ROI(CLIP1_SHAPE, CLIP1_VCOURT_RAND_PT3['x'], CLIP1_VCOURT_RAND_PT3['y'], CLIP1_VCOURT_RAND_PT3['w'], CLIP1_VCOURT_RAND_PT3['h'])
	# ROI_CLIP1_RAND_PT4 = generate_ROI(CLIP1_SHAPE, CLIP1_VCOURT_RAND_PT4['x'], CLIP1_VCOURT_RAND_PT4['y'], CLIP1_VCOURT_RAND_PT4['w'], CLIP1_VCOURT_RAND_PT4['h'])
	# ROI_CLIP1_RAND_PT5 = generate_ROI(CLIP1_SHAPE, CLIP1_VCOURT_RAND_PT5['x'], CLIP1_VCOURT_RAND_PT5['y'], CLIP1_VCOURT_RAND_PT5['w'], CLIP1_VCOURT_RAND_PT5['h'])

	# clip1_vcourt_pt1 = motion_tracking(CLIP1, ROI_CLIP1_RAND_PT1, start=0, end=630, maxCorners=1)
	# clip1_vcourt_pt2 = motion_tracking(CLIP1, ROI_CLIP1_RAND_PT2, start=0, end=630, maxCorners=1)
	# clip1_vcourt_pt3 = motion_tracking(CLIP1, ROI_CLIP1_RAND_PT3, start=0, end=630, maxCorners=1)
	# clip1_vcourt_pt4 = motion_tracking(CLIP1, ROI_CLIP1_RAND_PT4, start=0, end=630, maxCorners=1)
	# clip1_vcourt_pt5 = motion_tracking(CLIP1, ROI_CLIP1_RAND_PT5, start=0, end=630, maxCorners=1)

	# np.savetxt('./clip1_vcourt_br.txt', clip1_vcourt_pt1)
	# np.savetxt('./clip1_vcourt_nl.txt', clip1_vcourt_pt2)
	# np.savetxt('./clip1_vcourt_nr.txt', clip1_vcourt_pt3)
	# np.savetxt('./clip1_vcourt_rm.txt', clip1_vcourt_pt4)
	# np.savetxt('./clip1_vcourt_lm.txt', clip1_vcourt_pt5)

	# Load u,v coordinates of 5 points on the plane
	clip1_vcourt_pt1 = np.loadtxt('./clip1_vcourt_br.txt')
	clip1_vcourt_pt2 = np.loadtxt('./clip1_vcourt_nl.txt')
	clip1_vcourt_pt3 = np.loadtxt('./clip1_vcourt_nr.txt')
	clip1_vcourt_pt4 = np.loadtxt('./clip1_vcourt_rm.txt')
	clip1_vcourt_pt5 = np.loadtxt('./clip1_vcourt_lm.txt')
	# print(clip1_vcourt_pt1[0,:])
	# clip1_p1_feet   = np.loadtxt('./clip1_p1_feet.txt')

	# For clip1, use frame 1 as reference frame
	clip1_br_ref = np.hstack((clip1_vcourt_pt1[0,:], 1))
	clip1_nl_ref = np.hstack((clip1_vcourt_pt2[0,:], 1))
	clip1_nr_ref = np.hstack((clip1_vcourt_pt3[0,:], 1))
	clip1_rm_ref = np.hstack((clip1_vcourt_pt4[0,:], 1))
	clip1_lm_ref = np.hstack((clip1_vcourt_pt5[0,:], 1))


	# Calculate homography between camera-i and camera-0 where i represents frame number
	# ref_frame = np.vstack((
	# 	clip1_br_ref,
	# 	clip1_nl_ref,
	# 	clip1_nr_ref,
	# 	clip1_rm_ref,
	# 	clip1_lm_ref
	# ))
	# eigenvalues = []
	H = np.zeros((3,3))
	srcPts = np.vstack((
		clip1_vcourt_pt1[300,:],
		clip1_vcourt_pt2[300,:],
		clip1_vcourt_pt3[300,:],
		clip1_vcourt_pt4[300,:],
		clip1_vcourt_pt5[300,:]
	))
	for i in range(1, clip1_vcourt_pt1.shape[0]):
		dstPts = np.vstack((
			clip1_vcourt_pt1[i,:],
			clip1_vcourt_pt2[i,:],
			clip1_vcourt_pt3[i,:],
			clip1_vcourt_pt4[i,:],
			clip1_vcourt_pt5[i,:]
		))
		h, mask = cv2.findHomography(dstPts, srcPts, cv2.RANSAC, 5.0)
		H = np.vstack((H, h))
	H = H[3:]
	# print(H)


	# Piece all [u,v] together back to reference frame
	PAN_WIDTH = 630
	PAN_HEIGHT = 300
	count = 0
	plt.figure()
	avgImg = np.zeros((PAN_HEIGHT,PAN_WIDTH,3), dtype=np.float32)
	new_img = np.full((PAN_HEIGHT,PAN_WIDTH,3), 255, dtype='uint8')
	normImg = np.zeros((PAN_HEIGHT,PAN_WIDTH,3), dtype='uint8')
	try:
		# Initialize video writer and codecs
		cap = cv2.VideoCapture(filename)
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(filename.split('/')[2].split('.')[0] + "_panorama.avi", fourcc, 60.0, (PAN_WIDTH, PAN_HEIGHT), True)
		while(cap.isOpened() and count < end):
			ret, frame = cap.read()
			if count < skip:
				count +=1
				continue
			count += 1
			print("Frame : %d " % count)
			cv2.imshow("Original", frame)
			stacked_h = H[3*count:3*count+3]
			alpha = 1.0 / count
			new_img = cv2.warpPerspective(frame, stacked_h, (PAN_WIDTH, PAN_HEIGHT))
			new_img = cv2.convertScaleAbs(new_img)
			# new_img = extendBorder(new_img)rgb(235,221,192)
			new_img = colorBackground(new_img, (192,221,235)) # Sand color
			# cv2.accumulateWeighted(new_img, avgImg, alpha)
			# normImg = cv2.convertScaleAbs(avgImg)
			# Write processed frame back to video file
			writer.write(new_img)
			cv2.imshow("Stitched", new_img)
			key = cv2.waitKey(1) & 0xFF 
			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				break
	except Exception as e:
		print e
	finally:
		cv2.imwrite('bg.jpg', normImg)
		cap.release()
		writer.release()
		cv2.destroyAllWindows()

def colorBackground(img, color):
	""" color black background with a single constant RBG color given by color = (B, G, R)"""
	(height, width, channel) = img.shape
	tol = 50 # Threshold below which we treat it as black
	start_left = int(width/4)
	start_right = int(2*width/4)
	start_top = int(height/4)
	start_bot = int(2*height/4)
	for u in range(0, start_left):
		for v in range(0, height):
			if np.sum(img[v,u,:]) < tol:
				img[v,u,:] = np.array([color[0],color[1],color[2]], dtype='uint8')
	for u in range(start_right, width):
		for v in range(0, height):
			if np.sum(img[v,u,:]) < tol:
				img[v,u,:] = np.array([color[0],color[1],color[2]], dtype='uint8')
	for u in range(start_left, start_right):
		for v in range(0, start_top):
			if np.sum(img[v,u,:]) < tol:
				img[v,u,:] = np.array([color[0],color[1],color[2]], dtype='uint8')
	for u in range(start_left, start_right):
		for v in range(start_bot, height):
			if np.sum(img[v,u,:]) < tol:
				img[v,u,:] = np.array([color[0],color[1],color[2]], dtype='uint8')
	return img

def extendBorder(img, up=False, down=False, left=False, right=False):
	""" Fill zeros with last border"""
	(height, width, channel) = img.shape
	# Start from center, traverse column wise
	center_u = int(width/2)
	center_v = int(height/2)
	for u in range(0, width):
		last_top_border = np.array([0,0,0], dtype='uint8')
		last_bot_border = np.array([0,0,0], dtype='uint8')
		top_v = center_v
		bot_v = center_v
		for v in range(0, int(math.floor(-height/2)), -1):
			if np.sum(img[center_v + v, u, :]) == 0:
				top_v = v
				break
			else:
				last_top_border = img[center_v + v, u, :]
		if top_v != center_v:
			for v in range(top_v, int(math.floor(-height/2)), -1):
				img[center_v + v, u, :] = last_top_border
		for v in range(0, int(math.floor(height/2))):
			if np.sum(img[center_v + v, u, :]) == 0:
				bot_v = v
				break
			else:
				last_bot_border = img[center_v + v, u, :]
		if bot_v != center_v:
			for v in range(bot_v, int(math.floor(height/2))):
				img[center_v + v, u, :] = last_bot_border
	return img

def addPlayersToBackground(filename):
	PAN_WIDTH  = 630
	PAN_HEIGHT = 300
	cap  = cv2.VideoCapture(filename)
	bg   = cv2.imread('bg.jpg',  cv2.IMREAD_COLOR)
	# Median Filtering
	bg   = cv2.medianBlur(bg, 5)
	cv2.imshow("Median Filtering", bg)
	
	# Initialize resources
	fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=100, detectShadows=False)
	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
	writer = cv2.VideoWriter(filename.split('/')[1].split('.')[0] + "_with_players.avi", fourcc, 60.0, (PAN_WIDTH, PAN_HEIGHT), True)

	# Histogram equalization
	# bg_yuv = cv2.cvtColor(bg, cv2.COLOR_BGR2YUV)
	# bg_yuv[:,:,0] = cv2.equalizeHist(bg_yuv[:,:,0])
	# bg = cv2.cvtColor(bg_yuv, cv2.COLOR_YUV2BGR)
	# cv2.imshow("Histogram equalized background", bg)
	while(1):
		ret, frame = cap.read()
		fgmask = fgbg.apply(frame)
		bgmask = cv2.bitwise_not(fgmask)
		foreground = cv2.bitwise_and(frame, frame, mask=fgmask)
		background = cv2.bitwise_and(bg, bg, mask=bgmask)
		combined = cv2.add(foreground, background)
		cv2.imshow('frame', combined)
		writer.write(combined)
		k = cv2.waitKey(30) & 0xff
		if k == 27:
 			break
	cap.release()
	writer.release()
	cv2.destroyAllWindows()

def main():
	# Specify regions of interest for tracking objects
	# show_frame_in_matplot(CLIP1, 0)
	ROI_CLIP1_VCOURT_BR = generate_ROI(CLIP1_SHAPE, CLIP1_VCOURT_BOT_RIGHT['x'], CLIP1_VCOURT_BOT_RIGHT['y'], CLIP1_VCOURT_BOT_RIGHT['w'], CLIP1_VCOURT_BOT_RIGHT['h'])
	ROI_CLIP1_VCOURT_NR = generate_ROI(CLIP1_SHAPE, CLIP1_VCOURT_NET_RIGHT['x'], CLIP1_VCOURT_NET_RIGHT['y'], CLIP1_VCOURT_NET_RIGHT['w'], CLIP1_VCOURT_NET_RIGHT['h'])
	ROI_CLIP1_VCOURT_NL = generate_ROI(CLIP1_SHAPE, CLIP1_VCOURT_NET_LEFT['x'], CLIP1_VCOURT_NET_LEFT['y'], CLIP1_VCOURT_NET_LEFT['w'], CLIP1_VCOURT_NET_LEFT['h'])
	ROI_CLIP1_VCOURT_RM = generate_ROI(CLIP1_SHAPE, CLIP1_VCOURT_RIGHT_MID['x'], CLIP1_VCOURT_RIGHT_MID['y'], CLIP1_VCOURT_RIGHT_MID['w'], CLIP1_VCOURT_RIGHT_MID['h'])
	ROI_CLIP1_VCOURT_LM = generate_ROI(CLIP1_SHAPE, CLIP1_VCOURT_LEFT_MID['x'], CLIP1_VCOURT_LEFT_MID['y'], CLIP1_VCOURT_LEFT_MID['w'], CLIP1_VCOURT_LEFT_MID['h'])

	# Track image coordinates of known points on the floor plane
	# clip1_vcourt_br = motion_tracking(CLIP1, ROI_CLIP1_VCOURT_BR, start=0, end=630, maxCorners=1, skip=(240,280))
	# clip1_vcourt_nl = motion_tracking(CLIP1, ROI_CLIP1_VCOURT_NL, start=0, end=630, maxCorners=1)
	# clip1_vcourt_nr = motion_tracking(CLIP1, ROI_CLIP1_VCOURT_NR, start=0, end=630, maxCorners=1)
	# clip1_vcourt_rm = motion_tracking(CLIP1, ROI_CLIP1_VCOURT_RM, start=0, end=630, maxCorners=1)
	# clip1_vcourt_lm = motion_tracking(CLIP1, ROI_CLIP1_VCOURT_LM, start=0, end=630, maxCorners=1)
	
	# np.savetxt('./clip1_vcourt_br.txt', clip1_vcourt_br)
	# np.savetxt('./clip1_vcourt_nl.txt', clip1_vcourt_nl)
	# np.savetxt('./clip1_vcourt_nr.txt', clip1_vcourt_nr)
	# np.savetxt('./clip1_vcourt_rm.txt', clip1_vcourt_rm)
	# np.savetxt('./clip1_vcourt_lm.txt', clip1_vcourt_lm)

	# Make sure they are of right dimensions
	# print(len(clip1_vcourt_br))
	# print(len(clip1_vcourt_nl))
	# print(len(clip1_vcourt_nr))
	# print(len(clip1_vcourt_rm))
	# print(len(clip1_vcourt_lm))
	
	# Track players
	# ROI_CLIP1_GREEN_P1 = generate_ROI(CLIP1_SHAPE, CLIP1_GREEN_P1_ROI['x'], CLIP1_GREEN_P1_ROI['y'], CLIP1_GREEN_P1_ROI['w'], CLIP1_GREEN_P1_ROI['h'])
	# clip1_p1_feet = motion_tracking(CLIP1, ROI_CLIP1_GREEN_P1, start=360, end=630, maxCorners=3)
	# clip1_p1_feet = clip1_p1_feet[1::2,:]
	# print(clip1_p1_feet.shape)
	# np.savetxt('./clip1_p1_feet.txt', clip1_p1_feet)

	# ROI_CLIP1_GREEN_P2 = generate_ROI(CLIP1_SHAPE, CLIP1_GREEN_P2_ROI['x'], CLIP1_GREEN_P2_ROI['y'], CLIP1_GREEN_P2_ROI['w'], CLIP1_GREEN_P2_ROI['h'])
	# clip1_p2_feet = motion_tracking(CLIP1, ROI_CLIP1_GREEN_P2, start=0, end=720, maxCorners=1)
	# print(clip1_p2_feet) # Take 1st corner which is left knee

	# pts = np.vstack((
	# 	VCOURT_BOT_RIGHT,
	# 	VCOURT_NET_LEFT,
	# 	VCOURT_NET_RIGHT,
	# 	VCOURT_RIGHT_MID,
	# 	VCOURT_LEFT_MID
	# 	))
	# H = np.zeros((3,3))

	# clip1_vcourt_br = np.loadtxt('./clip1_vcourt_br.txt')
	# clip1_vcourt_nl = np.loadtxt('./clip1_vcourt_nl.txt')
	# clip1_vcourt_nr = np.loadtxt('./clip1_vcourt_nr.txt')
	# clip1_vcourt_rm = np.loadtxt('./clip1_vcourt_rm.txt')
	# clip1_vcourt_lm = np.loadtxt('./clip1_vcourt_lm.txt')
	# clip1_p1_feet   = np.loadtxt('./clip1_p1_feet.txt')
	# Get homography matrix for each frame
	# for i in range(0, clip1_vcourt_br.shape[0]): # i is frame number
	# eigenvalues = []
	# for i in range(0, clip1_p1_feet.shape[0]):
	# 	cam = np.vstack((
	# 		clip1_vcourt_br[i,:], 
	# 		clip1_vcourt_nl[i,:],
	# 		clip1_vcourt_nr[i,:],
	# 		clip1_vcourt_rm[i,:],
	# 		clip1_vcourt_lm[i,:]))
	# 	(h, s) = calc_homography(pts, cam)
	# 	eigenvalues.append(s)
	# 	H = np.vstack((H, h))
	# H = H[3:] # Discard first frame of 0s
	# # Average error/noise in our Homography matrices
	# avg = sum(eigenvalues) / len(eigenvalues)
	# print("Averge error is %.2f: " % avg)

	# Get plane coordinates for each player position in the image
	# PLAYER_COORDS = np.zeros((1,3))
	# for i in range(0, clip1_p1_feet.shape[0]):
	# 	coord = get_plane_coordinates(H[3*i:3*i+3], np.hstack((clip1_p1_feet[i,:],1)))
	# 	PLAYER_COORDS = np.vstack((PLAYER_COORDS, coord))
	# np.savetxt('./player.txt', PLAYER_COORDS[1:,:])
	# pts = np.loadtxt('./player.txt')
	# plot_player(pts)

	# Verify H by getting back reference pts in plane coordinates
	# REF_COORDS = np.zeros((1,3))
	# for i in range(0, clip1_p1_feet.shape[0]):
	# 	coord = get_plane_coordinates(H[3*i:3*i+3], np.hstack((clip1_vcourt_br[i,:],1)))
	# 	REF_COORDS = np.vstack((REF_COORDS, coord))
	# print(REF_COORDS)
	# plot_player(REF_COORDS)

	# for i in range(0, clip1_p1_feet.shape[0]):
	# 	coord = get_plane_coordinates(H[3*i:3*i+3], np.hstack((clip1_vcourt_nl[i,:],1)))
	# 	REF_COORDS = np.vstack((REF_COORDS, coord))
	# print(REF_COORDS)
	# plot_player(REF_COORDS[1:])

	# for i in range(0, clip1_p1_feet.shape[0]):
	# 	coord = get_plane_coordinates(H[3*i:3*i+3], np.hstack((clip1_vcourt_nr[i,:],1)))
	# 	REF_COORDS = np.vstack((REF_COORDS, coord))
	# print(REF_COORDS)
	# plot_player(REF_COORDS[1:])

	# for i in range(0, clip1_p1_feet.shape[0]):
	# 	coord = get_plane_coordinates(H[3*i:3*i+3], np.hstack((clip1_vcourt_rm[i,:],1)))
	# 	REF_COORDS = np.vstack((REF_COORDS, coord))
	# print(REF_COORDS)
	# plot_player(REF_COORDS[1:])

	# for i in range(0, clip1_p1_feet.shape[0]):
	# 	coord = get_plane_coordinates(H[3*i:3*i+3], np.hstack((clip1_vcourt_lm[i,:],1)))
	# 	REF_COORDS = np.vstack((REF_COORDS, coord))
	# print(REF_COORDS)
	# plot_player(REF_COORDS[1:])

	constructPanorama(CLIP1, 0, 630)
	# addPlayersToBackground(CLIP1_PAN)
	# bg = get_bg(CLIP1_PAN)
	# bg_yuv = cv2.cvtColor(bg, cv2.COLOR_BGR2YUV)
	# bg_yuv[:,:,0] = cv2.equalizeHist(bg_yuv[:,:,0])
	# bg = cv2.cvtColor(bg_yuv, cv2.COLOR_YUV2BGR)
	# cv2.imshow("Histogram equalized background", bg)
	# cv2.waitKey(0)

if __name__ == "__main__":
	main()
