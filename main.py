from __future__ import division # All division is floating point
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

version_flag = 2
if is_cv2():
	version_flag = 2
	import cv2.cv as cv
elif is_cv3():
	version_flag = 3

# cv2.ocl.setUseOpenCL(False)

# Acknowledgements - The team would like to acknowledge the following resources referenced in our project
# Processing video frames and writing to video file: http://www.pyimagesearch.com/2016/02/22/writing-to-video-with-opencv/
# Lucas-Kanade optical flow tracker -

# We define world origin as (0,0,0) to be at the center of the volleyball court.
# The volleyball court floor is used as the plane for homography calculation purposes
# Hence, because of our choice of coordinates, the world coordinates coincide with our plane coordinates
# with (Wx, Wy, Wz) = (Vp, Up, Zp)

# Using 1 cm = 1 unit as our scale, we can write down the coordinates of 5 key points on the plane
# For purposes of feature extraction, we need to identify points that are good corners and appear consistently
# in all frames of the clip
# Size of olympic sized beach volleyball court - 8m wide by 16m long
VCOURT_CENTER    = np.array([0,0])
VCOURT_TOP_LEFT  = np.array([-400, -800])
VCOURT_TOP_RIGHT = np.array([400, -800])
VCOURT_BOT_LEFT  = np.array([-400, 800])
VCOURT_BOT_RIGHT = np.array([400, 800])
VCOURT_NET_LEFT  = np.array([-500,0])
VCOURT_NET_RIGHT = np.array([500,0])
VCOURT_LEFT_MID  = np.array([-400,0])
VCOURT_RIGHT_MID = np.array([400,0])
VCOURT_BOT_MID   = np.array([0,800])
VCOURT_TOP_MID   = np.array([0,-800])

def get_bg(clip, repeat=None):
	""" 
		Get background of image by averaging method. Only works for stationary camera and background
		Repeat = list of tuples e.g. [(start, stop), (start,stop)] where we will add all the frames from start to stop again
	"""
	filename = PANORAMA_ROI[clip]['panorama_filename']
	cap = cv2.VideoCapture(filename)
	fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = int(cap.get(cv2.CAP_PROP_FPS))
	fc  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	ret, frame = cap.read()
	count  = 1
	avgImg = np.zeros(frame.shape, dtype=np.float32)
	if repeat is not None:
		curr_repeat = repeat.pop(0)
		curr_start = curr_repeat[0]
		curr_end   = curr_repeat[1]
	else:
		curr_repeat = None
		curr_start  = None
		curr_end    = None
	while(cap.isOpened() and ret):
		alpha = 1.0 / count
		if curr_start is not None and count >= curr_start and curr_end is not None and curr_end > count:
			print("Double counting frame %d" % count)
			cv2.accumulateWeighted(frame, avgImg, alpha)
			cv2.accumulateWeighted(frame, avgImg, alpha)
		cv2.accumulateWeighted(frame, avgImg, alpha)
		print("Frame = " + str(count) + ", alpha = " + str(alpha))
		cv2.imshow('background', avgImg)
		cv2.waitKey(1)
		normImg = cv2.convertScaleAbs(2.5 * avgImg)
		cv2.waitKey(1)
		cv2.imshow('normalized background', normImg)
		ret, frame = cap.read()
		count += 1
		if curr_start is not None and curr_start < count and curr_end is not None and curr_end <= count:
			if len(repeat) > 0:
				curr_repeat = repeat.pop(0)
				curr_start = curr_repeat[0]
				curr_end   = curr_repeat[1]
	print("Frame width       : %d" % fw)
	print("Frame height      : %d" % fh)
	print("Frames per second : %d" % fps)
	print("Frame count       : %d" % fc)
	cv2.imwrite('.' + filename.split('.')[1] + '_bg.jpg', normImg)
	cap.release()
	cv2.destroyAllWindows()
	return normImg

def show_frame_in_matplot(filename, num, roi=None):
	""" Show frame number from filename in matplot"""
	cap = cv2.VideoCapture(filename)
	count = 0
	while(cap.isOpened() and count <= num):
		count += 1
		ret, frame = cap.read()
	if roi is not None:
		top_left = (roi['x'], roi['y'])
		bottom_right = (roi['x']+roi['w'], roi['y']+roi['h'])
		frame = cv2.rectangle(frame, top_left, bottom_right, (0,0,255), 3)	
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
	plt.xlim([-500,500])
	plt.ylim([-1000, 1000])
	for i in range(0, pts.shape[0]):
		# if pts[i,2] > -0.8 and pts[i,2] < 1.2:
		plt.scatter(pts[i,0], pts[i,1], marker='x')
		plt.show()
		plt.pause(0.016)
	raw_input()

def constructPanorama(clip):
	"""
	 	clip is one of [clip1, clip2, clip3, clip4, clip5, clip6, clip7]
	 	ref is frame number to set as reference frame
	"""
	clip_num       = int(clip[-1])
	ref_frame      = PANORAMA_ROI[clip]['ref_frame']
	start_frame    = PANORAMA_ROI[clip]['start_frame']
	end_frame      = PANORAMA_ROI[clip]['end_frame']
	filename       = PANORAMA_ROI[clip]['filename']
	original_shape = PANORAMA_ROI[clip]['original_shape']
	pan_shape      = PANORAMA_ROI[clip]['panorama_shape']
	pt1            = PANORAMA_ROI[clip]['pt1']
	pt2            = PANORAMA_ROI[clip]['pt2']
	pt3            = PANORAMA_ROI[clip]['pt3']
	pt4            = PANORAMA_ROI[clip]['pt4']
	pt5            = PANORAMA_ROI[clip]['pt5']

	pt1_roi = generate_ROI(original_shape, pt1['x'], pt1['y'], pt1['w'], pt1['h'])
	pt2_roi = generate_ROI(original_shape, pt2['x'], pt2['y'], pt2['w'], pt2['h'])
	pt3_roi = generate_ROI(original_shape, pt3['x'], pt3['y'], pt3['w'], pt3['h'])
	pt4_roi = generate_ROI(original_shape, pt4['x'], pt4['y'], pt4['w'], pt4['h'])
	pt5_roi = generate_ROI(original_shape, pt5['x'], pt5['y'], pt5['w'], pt5['h'])

###################################################################################################
# Run Once to get txt for player position and track points, load the txt file for subsequent  
###################################################################################################

	# vcourt_pt1 = motion_tracking(filename, pt1_roi, start=start_frame, end=end_frame, maxCorners=1)
	# vcourt_pt2 = motion_tracking(filename, pt2_roi, start=start_frame, end=end_frame, maxCorners=1)
	# vcourt_pt3 = motion_tracking(filename, pt3_roi, start=start_frame, end=end_frame, maxCorners=1)
	# vcourt_pt4 = motion_tracking(filename, pt4_roi, start=start_frame, end=end_frame, maxCorners=1)
	# vcourt_pt5 = motion_tracking(filename, pt5_roi, start=start_frame, end=end_frame, maxCorners=1)

	# np.savetxt('./clip%d_vcourt_pt%d.txt' % (clip_num, 1), vcourt_pt1)
	# np.savetxt('./clip%d_vcourt_pt%d.txt' % (clip_num, 2), vcourt_pt2)
	# np.savetxt('./clip%d_vcourt_pt%d.txt' % (clip_num, 3), vcourt_pt3)
	# np.savetxt('./clip%d_vcourt_pt%d.txt' % (clip_num, 4), vcourt_pt4)
	# np.savetxt('./clip%d_vcourt_pt%d.txt' % (clip_num, 5), vcourt_pt5)

###################################################################################################

	# Load u,v coordinates of 5 points on the plane
	vcourt_pt1 = np.loadtxt('./clip%d_vcourt_pt%d.txt' % (clip_num, 1,))
	vcourt_pt2 = np.loadtxt('./clip%d_vcourt_pt%d.txt' % (clip_num, 2,))
	vcourt_pt3 = np.loadtxt('./clip%d_vcourt_pt%d.txt' % (clip_num, 3,))
	vcourt_pt4 = np.loadtxt('./clip%d_vcourt_pt%d.txt' % (clip_num, 4,))
	vcourt_pt5 = np.loadtxt('./clip%d_vcourt_pt%d.txt' % (clip_num, 5,))

	# For clip1, use frame 300 as reference frame
	H = np.zeros((3,3))
	dstPts = np.vstack((
		vcourt_pt1[ref_frame,:],
		vcourt_pt2[ref_frame,:],
		vcourt_pt3[ref_frame,:],
		vcourt_pt4[ref_frame,:],
		vcourt_pt5[ref_frame,:]
	))
	# Calculate homography between camera-i and camera-0 where i represents frame number
	for i in range(1, vcourt_pt1.shape[0]):
		srcPts = np.vstack((
			vcourt_pt1[i,:],
			vcourt_pt2[i,:],
			vcourt_pt3[i,:],
			vcourt_pt4[i,:],
			vcourt_pt5[i,:]
		))
		h, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5.0)
		H = np.vstack((H, h))
	H = H[3:]
	# print(H)

	# Piece all [u,v] together back to reference frame
	PAN_WIDTH = pan_shape[0]
	PAN_HEIGHT = pan_shape[1]
	count = 0
	plt.figure()
	new_img = np.full((PAN_HEIGHT,PAN_WIDTH,3), 255, dtype='uint8')
	try:
		# Initialize video writer and codecs
		cap = cv2.VideoCapture(filename)
		if version_flag == 3:
			fourcc = cv2.VideoWriter_fourcc(*"mp4v")
			writer = cv2.VideoWriter(filename.split('/')[2].split('.')[0] + "_panorama.mov", fourcc, 60.0, (PAN_WIDTH, PAN_HEIGHT), True)
		elif version_flag == 2:
			fourcc = cv.CV_FOURCC('m','p','4','v')
			writer = cv2.VideoWriter(filename.split('/')[2].split('.')[0] + "_panorama.mov", fourcc, 24, (PAN_WIDTH, PAN_HEIGHT), True)
		while(cap.isOpened() and count < end_frame-1):
			ret, frame = cap.read()
			if count < start_frame:
				count +=1
				continue
			count += 1
			print("Frame : %d " % count)
			cv2.imshow("Original", frame)
			stacked_h = H[3*count:3*count+3]
			new_img = cv2.warpPerspective(frame, stacked_h, (PAN_WIDTH, PAN_HEIGHT))
			new_img = cv2.convertScaleAbs(new_img)
			# new_img = extendBorder(new_img)rgb(235,221,192)
			# new_img = colorBackground(new_img, (192,221,235)) # Sand color
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
		cap.release()
		writer.release()
		cv2.destroyAllWindows()

def mergePanWithBg(clip):
	"""
		Merge panorama intermediate video with averaged background
	"""
	pan_shape      = PANORAMA_ROI[clip]['panorama_shape']
	start_frame    = PANORAMA_ROI[clip]['start_frame']
	end_frame      = PANORAMA_ROI[clip]['end_frame']

	PAN_WIDTH  = pan_shape[0]
	PAN_HEIGHT = pan_shape[1]
	count = 0
	bg = cv2.imread(PANORAMA_ROI[clip]['panorama_bg_filename'], cv2.IMREAD_COLOR)
	videoFile = PANORAMA_ROI[clip]['panorama_filename']
	cap = cv2.VideoCapture(videoFile)
	if version_flag == 3:
		fourcc = cv2.VideoWriter_fourcc(*"mp4v")
		writer = cv2.VideoWriter(filename.split('/')[2].split('.')[0] + "_panorama.mov", fourcc, 60.0, (PAN_WIDTH, PAN_HEIGHT), True)
	elif version_flag == 2:
		fourcc = cv.CV_FOURCC('m','p','4','v')
		writer = cv2.VideoWriter(filename.split('/')[2].split('.')[0] + "_panorama.mov", fourcc, 24, (PAN_WIDTH, PAN_HEIGHT), True)
	tol = 50
	while(cap.isOpened() and count < end_frame):
		ret, frame = cap.read()
		if count < start_frame:
			count += 1
			continue
		count += 1
		print("Frame : %d" % count)
		fgmask = np.zeros((frame.shape[0],frame.shape[1]), dtype='uint8')
		fgmask[np.where(np.sum(frame, axis=2) >= tol)] = 255
		bgmask = np.zeros((frame.shape[0],frame.shape[1]), dtype='uint8')
		bgmask[np.where(np.sum(frame, axis=2) < tol)] = 255
		foreground = cv2.bitwise_and(frame, frame, mask=fgmask)
		background = cv2.bitwise_and(bg, bg, mask=bgmask)
		combined = cv2.add(foreground, background)
		combined = cv2.convertScaleAbs(combined)
		cv2.imshow("Merged", combined)
		writer.write(combined)
		key = cv2.waitKey(1) & 0xFF
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
	cap.release()
	writer.release()
	cv2.destroyAllWindows()

def colorBackground(img, color):
	""" color black background with a single constant RBG color given by color = (B, G, R)"""
	(height, width, channel) = img.shape
	tol = 10 # Threshold below which we treat it as black
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

def subtractBackground(filename):
	cap  = cv2.VideoCapture(filename)
	# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
	fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=1000)

	while(1):
		ret, frame = cap.read()
		fgmask = fgbg.apply(frame)
		cv2.imshow('frame', fgmask)
		k = cv2.waitKey(30) & 0xff
		if k == 27:
 			break
	cap.release()
	cv2.destroyAllWindows()

def addPlayersToBackground(filename):
	PAN_WIDTH  = 630
	PAN_HEIGHT = 300
	cap  = cv2.VideoCapture(filename)
	bg   = cv2.imread(CLIP1_PAN_BG,  cv2.IMREAD_COLOR)
	# Median Filtering
	# bg   = cv2.medianBlur(bg, 3)
	# cv2.imshow("Median Filtering", bg)
	
	# Initialize resources
	fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=50, detectShadows=False)
	fourcc = cv2.VideoWriter_fourcc(*"mp4v")
	writer = cv2.VideoWriter(filename.split('/')[1].split('.')[0] + "_with_players.mov", fourcc, 60.0, (PAN_WIDTH, PAN_HEIGHT), True)
	ROI_mask = np.zeros((PAN_HEIGHT, PAN_WIDTH), dtype='uint8')
	ROI_mask[50:155, 70:480] = 255
	ROI_mask[100:250, 250:470] = 255
	while(1):
		ret, frame = cap.read()
		fgmask = fgbg.apply(frame)
		fgmask = cv2.bitwise_and(ROI_mask, ROI_mask, mask=fgmask)
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

mouseCoords = np.zeros((1000,2))
count = 0
def registerCoord(event, x, y, flags, param):
	global mouseCoords, count
	if event == cv2.EVENT_LBUTTONDOWN:
		mouseCoords[count] = np.array([x, y])
	elif event == cv2.EVENT_RBUTTONDOWN:
		mouseCoords[count] = np.array([10000, 10000])
	else:
		pass

def mouseMotionTracking(clip, obj, use_final=True):
	""" 
		Use this to track manually using mouse. Left click on player's feet. If player is out of view temporarily, right click
		clip is one of [clip1 - clip7]
		obj is one of [red1, red2, white1, white2, ball, green1, green2] - string
		use_final - default True, use the final panorama clip, else use the original panorama clip without background

		Here we follow this convention:
		red1   - means red team player who stands at the back of the court
		red2   - means red team player who stands at the front of the court
		green1 - means green team player who stands at the back of the court
		green2 - means green team player who stands at the front of the court
		white1 - means white team player who stands at the back of the court
		white2 - means white team player who stands at the front of the court
		ball   - the volleyball we are tracking
	"""
	global count, mouseCoords
	if use_final:
		filename = PANORAMA_ROI[clip]['panorama_final_filename']
	else:
		filename = PANORAMA_ROI[clip]['panorama_filename']
	end_frame = PANORAMA_ROI[clip]['player_%s_tracking_end_frame' % obj]
	start_frame = PANORAMA_ROI[clip]['player_%s_tracking_start_frame' % obj]
	cap = cv2.VideoCapture(filename)
	count = 0
	cv2.namedWindow('image')
	cv2.setMouseCallback('image', registerCoord)
	while(cap.isOpened() and count < end_frame):
		ret, frame = cap.read()
		if count < start_frame:
			count += 1
			continue
		count += 1
		print("Frame : %d" % count)
		if obj != 'ball':
			if count % 10 == 0:
				cv2.imshow("image", frame)
		else:
			cv2.imshow("image", frame)
		key = cv2.waitKey(50) & 0xFF
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
	cap.release()
	cv2.destroyAllWindows()

	# Post processing
	mouseCoords = fillZeros(mouseCoords)
	filename_key = 'player_%s_position_filename' % obj
	tracking_end_frame_key = 'player_%s_tracking_end_frame' % obj
	np.savetxt(PANORAMA_ROI[clip][filename_key], mouseCoords[0:int(PANORAMA_ROI[clip][tracking_end_frame_key]),:])

	# Clear mouseCoords and reset count
	mouseCoords = np.zeros((1000,2))
	count = 0

def fillZeros(arr):
	first_non_zero = np.array([0,0])
	prev_non_zero = np.array([0,0])
	is_set = False
	for i in range(0, arr.shape[0]):
		if np.sum(arr[i]) == 0:
			arr[i] = prev_non_zero
		else:
			if not is_set: first_non_zero = arr[i]
			is_set = True
			prev_non_zero = arr[i]
	for i in range(0, arr.shape[0]):
		if np.sum(arr[i]) == 0:
			arr[i] = first_non_zero
	return arr

def main():
	global mouseCoords
	clip = PANORAMA_ROI['clip1']
	show_frame_in_matplot(clip['panorama_filename'], 100)
	# print("Get ready to track green1")
	# time.sleep(2)
	# mouseMotionTracking('clip7', 'red1', use_final=False)
	# print("Get ready to track green2")
	# time.sleep(2)
	# mouseMotionTracking('clip7', 'red2', use_final=False)
	# print("Get ready to track white1")
	# time.sleep(2)
	# mouseMotionTracking('clip7', 'white1', use_final=False)
	# print("Get ready to track white2")
	# time.sleep(2)
	# mouseMotionTracking('clip7', 'white2', use_final=False)
	# print("Get ready to track ball")
	# time.sleep(2)
	# mouseMotionTracking('clip7', 'ball', use_final=False)
	
	# playerRed1_roi = generate_ROI(clip['panorama_roi_shape'], clip['playerRed1']['x'], clip['playerRed1']['y'], clip['playerRed1']['w'], clip['playerRed1']['h'])
	# playerRed1_pts = motion_tracking(clip['panorama_final_filename'], playerRed1_roi, start=clip['player_tracking_start_frame'], end=clip['player_tracking_end_frame'], maxCorners=10)
	# print(playerRed1_pts.shape)

	# Specify regions of interest for tracking objects
	# show_frame_in_matplot('./beachVolleyball1_panorama_with_players.mov', 0)

	# ROI_CLIP1_VCOURT_BR = generate_ROI(CLIP1_SHAPE, CLIP1_VCOURT_BOT_RIGHT['x'], CLIP1_VCOURT_BOT_RIGHT['y'], CLIP1_VCOURT_BOT_RIGHT['w'], CLIP1_VCOURT_BOT_RIGHT['h'])
	# ROI_CLIP1_VCOURT_NR = generate_ROI(CLIP1_SHAPE, CLIP1_VCOURT_NET_RIGHT['x'], CLIP1_VCOURT_NET_RIGHT['y'], CLIP1_VCOURT_NET_RIGHT['w'], CLIP1_VCOURT_NET_RIGHT['h'])
	# ROI_CLIP1_VCOURT_NL = generate_ROI(CLIP1_SHAPE, CLIP1_VCOURT_NET_LEFT['x'], CLIP1_VCOURT_NET_LEFT['y'], CLIP1_VCOURT_NET_LEFT['w'], CLIP1_VCOURT_NET_LEFT['h'])
	# ROI_CLIP1_VCOURT_RM = generate_ROI(CLIP1_SHAPE, CLIP1_VCOURT_RIGHT_MID['x'], CLIP1_VCOURT_RIGHT_MID['y'], CLIP1_VCOURT_RIGHT_MID['w'], CLIP1_VCOURT_RIGHT_MID['h'])
	# ROI_CLIP1_VCOURT_LM = generate_ROI(CLIP1_SHAPE, CLIP1_VCOURT_LEFT_MID['x'], CLIP1_VCOURT_LEFT_MID['y'], CLIP1_VCOURT_LEFT_MID['w'], CLIP1_VCOURT_LEFT_MID['h'])

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
	# print(clip1_p1_feet.shape)
	# print(clip1_vcourt_br.shape)
	# for i in range(0, 500):
	# 	cam = np.vstack((
	# 		clip1_vcourt_br[i,:], 
	# 		clip1_vcourt_nl[i,:],
	# 		clip1_vcourt_nr[i,:],
	# 		clip1_vcourt_rm[i,:],
	# 		clip1_vcourt_lm[i,:]))
	# 	h, mask = cv2.findHomography(pts, cam, cv2.RANSAC, 5.0)
	# 	if h is not None:
	# 		H = np.vstack((H, h))
	# 	else:
	# 		print(i)
	# H = H[3:] # Discard first frame of 0s
	
	# Get plane coordinates for each player position in the image
	# PLAYER_COORDS = np.zeros((1,3))
	# for i in range(360, clip1_p1_feet.shape[0]):
	# 	coord = get_plane_coordinates(H[3*i:3*i+3], np.hstack((clip1_p1_feet[i,:],1)))
	# 	PLAYER_COORDS = np.vstack((PLAYER_COORDS, coord))
	# np.savetxt('./player.txt', PLAYER_COORDS[1:,:])
	# pts = np.loadtxt('./player.txt')
	# plot_player(pts)

	# Verify H by getting back reference pts in plane coordinates
	# REF_COORDS = np.zeros((1,3))
	# for i in range(360, clip1_p1_feet.shape[0]):
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

	# constructPanorama('clip7')
	# bg = get_bg('clip6', repeat=[(300,400),(500,600)])
	# addPlayersToBackground(CLIP1_PAN)
	# mergePanWithBg('clip6')
	# subtractBackground(CLIP1_PAN)

if __name__ == "__main__":
	main()
