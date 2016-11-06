from __future__ import division
import numpy as np
import math
import numpy.linalg as la
import cv2
from homography import *
from corner import *
from roi import *
from imutils.video import VideoStream
import time
np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(threshold=np.inf)

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

def motion_tracking(filename, ROI, start=0, end=None, maxCorners=3, skip=None):
	""" 
		Uses OpenCV to extract good corners and track the movement of those corners.
		ROI is a mask to extract only features from this part of the image
		start from frame no., with 1st frame in video as frame 0
		end at frame no., with None defaulting to full video file
		Skip = (min, max) contains a range of frame numbers with which the algorithm should skip for better results
	"""
	cap = cv2.VideoCapture(filename)
	
	# params for ShiTomasi corner detection
	feature_params = dict( maxCorners = maxCorners,
						   qualityLevel = 0.1,
						   minDistance = 7,
						   blockSize = 7 )
	
	# Parameters for lucas kanade optical flow
	lk_params = dict(winSize  = (15,15),
					maxLevel = 7,
					criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
	
	# Create some random colors
	color = np.random.randint(0,255,(100,3))
	
	# Take first frame and find corners in it
	ret, old_frame = cap.read()
	
	old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
	p0 = cv2.goodFeaturesToTrack(old_gray, mask = ROI, **feature_params)
	
	# Create a mask image for drawing purposes
	mask = np.zeros_like(old_frame)
   
	count = 0
	results = np.zeros((1,2))
	while(1):

		ret,frame = cap.read()
		if count < start:
			count += 1
			continue
		else:
			count += 1

		if skip is not None and count >= skip[0] and count <= skip[1]:
			results = np.vstack((results, good_new))
			continue

		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# calculate optical flow
		p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

		# Select good points
		good_new = p1[st==1]
		good_old = p0[st==1]
	
		results = np.vstack((results, good_new))
		# draw the tracks
		for i,(new,old) in enumerate(zip(good_new,good_old)):
			a,b = new.ravel()
			c,d = old.ravel()
			mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
			frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
		img = cv2.add(frame,mask)
	
		cv2.imshow('frame',img)
		k = cv2.waitKey(1) & 0xff
		if k == 27:
			break
	
		# Now update the previous frame and previous points
		old_gray = frame_gray.copy()
		p0 = good_new.reshape(-1,1,2)

		if end is not None and count > end:
			break
	cv2.destroyAllWindows()
	cap.release()
	# print(results)
	return results[1:]

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

def calc_alpha():
	pass

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
	# for i in range(1, clip1_vcourt_pt1.shape[0]):
	# 	cam2 = np.vstack((
	# 		np.hstack((clip1_vcourt_pt1[i,:], 1)),
	# 		np.hstack((clip1_vcourt_pt2[i,:], 1)),
	# 		np.hstack((clip1_vcourt_pt3[i,:], 1)),
	# 		np.hstack((clip1_vcourt_pt4[i,:], 1)),
	# 		np.hstack((clip1_vcourt_pt5[i,:], 1))
	# 	))
	# 	(h, s) = calc_homography(ref_frame, cam2)
	# 	eigenvalues.append(s)
	# 	H = np.vstack((H, h))
	# H = H[3:] # Discard first frame of 0s
	# # Average error/noise in our Homography matrices
	# avg = sum(eigenvalues) / len(eigenvalues)
	# print("Averge error is %.5f: " % avg)
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

	# Initialize video writer and codecs
	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
	writer = cv2.VideoWriter(filename.split('/')[2].split('.')[0] + "_panorama.avi", fourcc, 60.0, (632, 300), True)

	# Piece all [u,v] together back to reference frame
	cap = cv2.VideoCapture(filename)
	count = 0
	plt.figure()
	NUM_PIXELS = 300 * 632
	stacked_u = np.zeros((3, NUM_PIXELS))
	index = 0
	offset_u = 100
	offset_v = 100
	u_scale = 632
	v_scale = 300
	for u2 in range(0, 632):
		for v2 in range(0, 300):
			stacked_u[0:3, index] = np.array([u2,v2,1])
			index += 1
	while(cap.isOpened() and count <= end):
		ret, frame = cap.read()
		cv2.imshow("Original", frame)
		if count < skip:
			count +=1
			continue
		new_img = np.full((300,632,3), 255, dtype='uint8')
		stacked_h = H[3*count:3*count+3]
		new_img = cv2.warpPerspective(frame, stacked_h, (632, 300))
		count += 1
		print("Frame : %d " % count)
		new_img = cv2.convertScaleAbs(new_img)
		# Write processed frame back to video file
		writer.write(new_img)
		cv2.imshow("Stitched", new_img)
		key = cv2.waitKey(1) & 0xFF 
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
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

	# get_bg(CLIP1)
	# cap = cv2.VideoCapture(CLIP1)
	# ret, frame = cap.read()
	# while cap.isOpened() and ret:
	# 	bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# 	# (corner, x, y) = harris_corner(bw)
	# 	cv2.imshow('background', frame)
	# 	cv2.waitKey(25)
	# 	ret, frame = cap.read()
	# cap.release()
	# cv2.destroyAllWindows()

	# constructPanorama(CLIP1, 0, 630)
	bg = get_bg(CLIP1_PAN)
	bg_yuv = cv2.cvtColor(bg, cv2.COLOR_BGR2YUV)
	bg_yuv[:,:,0] = cv2.equalizeHist(bg_yuv[:,:,0])
	bg = cv2.cvtColor(bg_yuv, cv2.COLOR_YUV2BGR)
	cv2.imshow("Histogram equalized background", bg)
	cv2.waitKey(0)

if __name__ == "__main__":
	main()
