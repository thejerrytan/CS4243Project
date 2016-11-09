import numpy as np
import math
import numpy.linalg as la
import cv2

PANORAMA_ROI = {
	'clip1' : {
		'filename'       : './beachVolleyball/beachVolleyball1.mov',
		'panorama_filename' : './beachVolleyball1_panorama.mov',
		'panorama_bg_filename' : './beachVolleyball1_panorama_bg.jpg',
		'start_frame'    : 0,
		'end_frame'      : 630,
		'original_shape' : (300, 632),
		'panorama_shape' : (630, 300),
		'ref_frame'      : 300,
		'pt1' : {
			'x' : 50,
			'y' : 220,
			'h' : 30,
			'w' : 30
		},
		'pt2' : {
			'x' : 50,
			'y' : 180,
			'h' : 30,
			'w' : 30
		},
		'pt3' : {
			'x' : 250,
			'y' : 200,
			'h' : 30,
			'w' : 30
		},
		'pt4' : {
			'x' : 280,
			'y' : 80,
			'h' : 30,
			'w' : 30
		},
		'pt5' : {
			'x' : 60,
			'y' : 150,
			'h' : 30,
			'w' : 30
		}
	},
	'clip2' : {
		'filename'       : './beachVolleyball/beachVolleyball2.mov',
		'panorama_filename' : './beachVolleyball2_panorama.mov',
		'panorama_bg_filename' : './beachVolleyball2_panorama_bg.jpg',
		'start_frame'    : 0,
		'end_frame'      : 550,
		'original_shape' : (296, 636),
		'panorama_shape' : (630, 300),
		'ref_frame'      : 100,
		'pt1' : {
			'x' : 510,
			'y' : 80,
			'h' : 50,
			'w' : 50
		},
		'pt2' : {
			'x' : 525,
			'y' : 120,
			'h' : 50,
			'w' : 50
		},
		'pt3' : {
			'x' : 550,
			'y' : 100,
			'h' : 30,
			'w' : 30
		},
		'pt4' : {
			'x' : 565,
			'y' : 90,
			'h' : 30,
			'w' : 30
		},
		'pt5' : {
			'x' : 580,
			'y' : 115,
			'h' : 30,
			'w' : 30
		}
	},
	'clip3' : {
		'filename'       : './beachVolleyball/beachVolleyball3.mov',
		'panorama_filename' : './beachVolleyball3_panorama.mov',
		'panorama_bg_filename' : './beachVolleyball3_panorama_bg.jpg',
		'start_frame'    : 0,
		'end_frame'      : 350,
		'original_shape' : (294, 636),
		'panorama_shape' : (700, 300),
		'ref_frame'      : 100,
		'pt1' : {
			'x' : 120,
			'y' : 150,
			'h' : 50,
			'w' : 50
		},
		'pt2' : {
			'x' : 250,
			'y' : 150,
			'h' : 50,
			'w' : 50
		},
		'pt3' : {
			'x' : 250,
			'y' : 260,
			'h' : 30,
			'w' : 30
		},
		'pt4' : {
			'x' : 120,
			'y' : 250,
			'h' : 20,
			'w' : 20
		},
		'pt5' : {
			'x' : 20,
			'y' : 120,
			'h' : 30,
			'w' : 30
		}
	},
	'clip4' : {
		'filename'       : './beachVolleyball/beachVolleyball4.mov',
		'panorama_filename' : './beachVolleyball4_panorama.mov',
		'panorama_bg_filename' : './beachVolleyball4_panorama_bg.jpg',
		'start_frame'    : 20,
		'end_frame'      : 600,
		'original_shape' : (296, 638),
		'panorama_shape' : (700, 300),
		'ref_frame'      : 100,
		'pt1' : {
			'x' : 600,
			'y' : 100,
			'h' : 100,
			'w' : 100
		},
		'pt2' : {
			'x' : 500,
			'y' : 130,
			'h' : 50,
			'w' : 50
		},
		'pt3' : {
			'x' : 450,
			'y' : 200,
			'h' : 30,
			'w' : 30
		},
		'pt4' : {
			'x' : 350,
			'y' : 150,
			'h' : 20,
			'w' : 20
		},
		'pt5' : {
			'x' : 400,
			'y' : 135,
			'h' : 30,
			'w' : 30
		}
	},
	'clip5' : {
		'filename'       : './beachVolleyball/beachVolleyball5.mov',
		'panorama_filename' : './beachVolleyball5_panorama.mov',
		'panorama_bg_filename' : './beachVolleyball5_panorama_bg.jpg',
		'start_frame'    : 0,
		'end_frame'      : 1000,
		'original_shape' : (354, 636),
		'panorama_shape' : (700, 354),
		'ref_frame'      : 500,
		'pt1' : {
			'x' : 460,
			'y' : 150,
			'h' : 100,
			'w' : 100
		},
		'pt2' : {
			'x' : 410,
			'y' : 170,
			'h' : 30,
			'w' : 30
		},
		'pt3' : {
			'x' : 490,
			'y' : 170,
			'h' : 30,
			'w' : 30
		},
		'pt4' : {
			'x' : 220,
			'y' : 180,
			'h' : 20,
			'w' : 20
		},
		'pt5' : {
			'x' : 200,
			'y' : 140,
			'h' : 30,
			'w' : 30
		}
	},
	'clip6' : {
		'filename'       : './beachVolleyball/beachVolleyball6.mov',
		'panorama_filename' : './beachVolleyball6_panorama.mov',
		'panorama_bg_filename' : './beachVolleyball6_panorama_bg.jpg',
		'start_frame'    : 0,
		'end_frame'      : 380,
		'original_shape' : (350, 636),
		'panorama_shape' : (700, 350),
		'ref_frame'      : 50,
		'pt1' : {
			'x' : 600,
			'y' : 100,
			'h' : 100,
			'w' : 100
		},
		'pt2' : {
			'x' : 330,
			'y' : 130,
			'h' : 50,
			'w' : 50
		},
		'pt3' : {
			'x' : 360,
			'y' : 230,
			'h' : 30,
			'w' : 30
		},
		'pt4' : {
			'x' : 420,
			'y' : 300,
			'h' : 20,
			'w' : 20
		},
		'pt5' : {
			'x' : 570,
			'y' : 270,
			'h' : 20,
			'w' : 20
		}
	},
	'clip7' : {
		'filename'          : './beachVolleyball/beachVolleyball7.mov',
		'panorama_filename' : './beachVolleyball7_panorama.mov',
		'panorama_bg_filename' : './beachVolleyball7_panorama_bg.jpg',
		'start_frame'       : 0,
		'end_frame'         : 1000,
		'original_shape'    : (356, 638),
		'panorama_shape'    : (700, 300),
		'ref_frame'         : 100,
		'pt1' : {
			'x' : 150,
			'y' : 140,
			'h' : 50,
			'w' : 50
		},
		'pt2' : {
			'x' : 150,
			'y' : 290,
			'h' : 50,
			'w' : 50
		},
		'pt3' : {
			'x' : 50,
			'y' : 170,
			'h' : 30,
			'w' : 30
		},
		'pt4' : {
			'x' : 150,
			'y' : 180,
			'h' : 20,
			'w' : 20
		},
		'pt5' : {
			'x' : 250,
			'y' : 150,
			'h' : 30,
			'w' : 30
		}
	}
}

CLIP1_GREEN_P1_ROI = {
	'x' : 330,
	'y' : 180,
	'h' : 50,
	'w' : 50
}
# CLIP1_GREEN_P1_ROI = {
# 	'x' : 470,
# 	'y' : 210,
# 	'h' : 30,
# 	'w' : 30
# }
CLIP1_GREEN_P2_ROI = {
	'x' : 190,
	'y' : 100,
	'h' : 50,
	'w' : 50
}

CLIP1_VCOURT_BOT_RIGHT = {
	'x' : 430,
	'y' : 130,
	'h' : 50,
	'w' : 50
}
CLIP1_VCOURT_BOT_LEFT = {
	'x' : 170,
	'y' : 275,
	'h' : 50,
	'w' : 50
}
CLIP1_VCOURT_NET_RIGHT = {
	'x' : 255,
	'y' : 70,
	'h' : 50,
	'w' : 50
}
CLIP1_VCOURT_NET_LEFT = {
	'x' : 35,
	'y' : 130,
	'h' : 50,
	'w' : 50
}
CLIP1_VCOURT_RIGHT_MID = {
	'x' : 280,
	'y' : 85,
	'h' : 10,
	'w' : 10
}
CLIP1_VCOURT_LEFT_MID = {
	'x' : 70,
	'y' : 130,
	'h' : 15,
	'w' : 15
}

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
	lk_params = dict(winSize  = (30,30),
					maxLevel = 7,
					criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
	
	# Create some random colors
	color = np.random.randint(0,255,(100,3))
	
	# Take first frame and find corners in it
	ret, old_frame = cap.read()
	print(old_frame.shape)
	old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
	p0 = cv2.goodFeaturesToTrack(old_gray, mask = ROI, **feature_params)
	
	# Create a mask image for drawing purposes
	mask = np.zeros_like(old_frame)
   
	count = 0
	results = np.zeros((1,2))
	while(1):

		ret,frame = cap.read()
		print("Frame : %d" % int(count))
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