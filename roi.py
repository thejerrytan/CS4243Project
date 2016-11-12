import numpy as np
import math
import numpy.linalg as la
import cv2

PLANE_COORDS = {
	'vcourt_center' : (0,0),
	'vcourt_top_left' : (-400,-800),
	'vcourt_top_right' : (400,-800),
	'vcourt_bottom_left': (-400,800),
	'vcourt_bottom_right' : (400,800),
	'vcourt_net_left' : (-500,0),
	'vcourt_net_right' : (500,0),
	'vcourt_left_mid' : (-400,0),
	'vcourt_right_mid' : (400,0),
	'vcourt_bot_mid' : (0,800),
    'vcourt_top_mid' : (0,-800),
    'vcourt_exp1' : (300, -100)
}

PANORAMA_ROI = {
	'clip1' : {
		'filename'       : './beachVolleyball/beachVolleyball1.mov',
		'panorama_filename' : './beachVolleyball1_panorama.mov',
		'panorama_bg_filename' : './beachVolleyball1_panorama_bg.jpg',
		'panorama_final_filename' : "./beachVolleyball1_panorama_final.mov",
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
		},
		'player_green1_tracking_end_frame' : 629,
		'player_green1_tracking_start_frame' : 0,
		'player_green1_position_filename' : './clip1_player_green1_position.txt',
		'player_green2_tracking_end_frame' : 629,
		'player_green2_tracking_start_frame' : 0,
		'player_green2_position_filename' : './clip1_player_green2_position.txt',
		'player_white1_tracking_end_frame' : 629,
		'player_white1_tracking_start_frame' : 0,
		'player_white1_position_filename' : './clip1_player_white1_position.txt',
		'player_white2_tracking_end_frame' : 629,
		'player_white2_tracking_start_frame' : 0,
		'player_white2_position_filename' : './clip1_player_white2_position.txt',
		'player_ball_tracking_end_frame' : 629,
		'player_ball_tracking_start_frame' : 0,
		'player_ball_position_filename' : './clip1_ball_position.txt',
		'vcourt_points' : ['vcourt_bottom_left','vcourt_bottom_right','vcourt_net_left','vcourt_net_right','vcourt_center'],
		'vcourt_bottom_left'  : (191,362),
		'vcourt_bottom_right' : (472, 159),
		'vcourt_net_left'     : (-50, 40), # HACKED POINT!!!
		'vcourt_net_right'    : (315, 96),
		'vcourt_center'       : (209, 117),
		'vcourt_top_left'     : (-1, -1),
		'vcourt_top_right'    : (-1, -1),
		'plane_homography_filename' : './clip1_plane_homography.txt',
		'player_green1_court_position_filename' : './clip1_player_green1_court_position.txt',
		'player_green2_court_position_filename' : './clip1_player_green2_court_position.txt',
		'player_white1_court_position_filename' : './clip1_player_white1_court_position.txt',
		'player_white2_court_position_filename' : './clip1_player_white2_court_position.txt',
		'player_ball_court_position_filename' : './clip1_ball_court_position.txt'
	},
	'clip2' : {
		'filename'       : './beachVolleyball/beachVolleyball2.mov',
		'panorama_filename' : './beachVolleyball2_panorama.mov',
		'panorama_bg_filename' : './beachVolleyball2_panorama_bg.jpg',
		'panorama_final_filename' : "./beachVolleyball2_panorama_final.mov",
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
		},
		'player_green1_tracking_end_frame' : 549,
		'player_green1_tracking_start_frame' : 0,
		'player_green1_position_filename' : './clip2_player_green1_position.txt',
		'player_green2_tracking_end_frame' : 549,
		'player_green2_tracking_start_frame' : 0,
		'player_green2_position_filename' : './clip2_player_green2_position.txt',
		'player_white1_tracking_end_frame' : 549,
		'player_white1_tracking_start_frame' : 0,
		'player_white1_position_filename' : './clip2_player_white1_position.txt',
		'player_white2_tracking_end_frame' : 549,
		'player_white2_tracking_start_frame' : 0,
		'player_white2_position_filename' : './clip2_player_white2_position.txt',
		'player_ball_tracking_end_frame' : 549,
		'player_ball_tracking_start_frame' : 0,
		'player_ball_position_filename' : './clip2_ball_position.txt',
		'vcourt_points' : ['vcourt_bottom_left','vcourt_bottom_right','vcourt_net_left','vcourt_net_right','vcourt_center'],
		'vcourt_bottom_left'  : (-1, -1),
		'vcourt_bottom_right' : (-1, -1),
		'vcourt_net_left'     : (176, 153),
		'vcourt_net_right'    : (607, 176),
		'vcourt_center'       : (-1, -1),
		'vcourt_top_left'     : (-1, -1),
		'vcourt_top_right'    : (545, 135),
		'plane_homography_filename' : './clip2_plane_homography.txt',
		'player_green1_court_position_filename' : './clip2_player_green1_court_position.txt',
		'player_green2_court_position_filename' : './clip2_player_green2_court_position.txt',
		'player_white1_court_position_filename' : './clip2_player_white1_court_position.txt',
		'player_white2_court_position_filename' : './clip2_player_white2_court_position.txt',
		'player_ball_court_position_filename' : './clip2_ball_court_position.txt'
	},
	'clip3' : {
		'filename'       : './beachVolleyball/beachVolleyball3.mov',
		'panorama_filename' : './beachVolleyball3_panorama.mov',
		'panorama_bg_filename' : './beachVolleyball3_panorama_bg.jpg',
		'panorama_final_filename' : "./beachVolleyball3_panorama_final.mov",
		'start_frame'    : 0,
		'end_frame'      : 350,
		'original_shape' : (294, 636),
		'panorama_shape' : (700, 300),
		'ref_frame'      : 50,
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
			'x' : 280,
			'y' : 250,
			'h' : 20,
			'w' : 20
		},
		'pt5' : {
			'x' : 20,
			'y' : 120,
			'h' : 30,
			'w' : 30
		},
		'player_green1_tracking_end_frame' : 349,
		'player_green1_tracking_start_frame' : 0,
		'player_green1_position_filename' : './clip3_player_green1_position.txt',
		'player_green2_tracking_end_frame' : 349,
		'player_green2_tracking_start_frame' : 0,
		'player_green2_position_filename' : './clip3_player_green2_position.txt',
		'player_white1_tracking_end_frame' : 349,
		'player_white1_tracking_start_frame' : 0,
		'player_white1_position_filename' : './clip3_player_white1_position.txt',
		'player_white2_tracking_end_frame' : 349,
		'player_white2_tracking_start_frame' : 0,
		'player_white2_position_filename' : './clip3_player_white2_position.txt',
		'player_ball_tracking_end_frame' : 349,
		'player_ball_tracking_start_frame' : 0,
		'player_ball_position_filename' : './clip3_ball_position.txt',
		'vcourt_points' : ['vcourt_bottom_left','vcourt_bottom_right','vcourt_net_right','vcourt_top_right'],
		'vcourt_bottom_left'  : (685, 305),
		'vcourt_bottom_right' : (690, 190),
		'vcourt_net_left'     : (-1, -1),
		'vcourt_net_right'    : (322, 180),
		'vcourt_center'       : (-1, -1),
		'vcourt_top_left'     : (-1, -1),
		'vcourt_top_right'    : (-50, 190),
		'vcourt_right_mid'    : (317, 190),
		'plane_homography_filename' : './clip3_plane_homography.txt',
		'player_green1_court_position_filename' : './clip3_player_green1_court_position.txt',
		'player_green2_court_position_filename' : './clip3_player_green2_court_position.txt',
		'player_white1_court_position_filename' : './clip3_player_white1_court_position.txt',
		'player_white2_court_position_filename' : './clip3_player_white2_court_position.txt',
		'player_ball_court_position_filename' : './clip3_ball_court_position.txt'
	},
	'clip4' : {
		'filename'       : './beachVolleyball/beachVolleyball4.mov',
		'panorama_filename' : './beachVolleyball4_panorama.mov',
		'panorama_bg_filename' : './beachVolleyball4_panorama_bg.jpg',
		'panorama_final_filename' : "./beachVolleyball4_panorama_final.mov",
		'start_frame'    : 20,
		'end_frame'      : 1000,
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
		},
		'player_green1_tracking_end_frame' : 999,
		'player_green1_tracking_start_frame' : 0,
		'player_green1_position_filename' : './clip4_player_green1_position.txt',
		'player_green2_tracking_end_frame' : 999,
		'player_green2_tracking_start_frame' : 0,
		'player_green2_position_filename' : './clip4_player_green2_position.txt',
		'player_white1_tracking_end_frame' : 999,
		'player_white1_tracking_start_frame' : 0,
		'player_white1_position_filename' : './clip4_player_white1_position.txt',
		'player_white2_tracking_end_frame' : 999,
		'player_white2_tracking_start_frame' : 0,
		'player_white2_position_filename' : './clip4_player_white2_position.txt',
		'player_ball_tracking_end_frame' : 999,
		'player_ball_tracking_start_frame' : 0,
		'player_ball_position_filename' : './clip4_ball_position.txt',
		'vcourt_points' : ['vcourt_bottom_left','vcourt_bottom_right','vcourt_net_left','vcourt_net_right','vcourt_center'],
		'vcourt_bottom_left'  : (-1, -1),
		'vcourt_bottom_right' : (-1, -1),
		'vcourt_net_left'     : (-1, -1),
		'vcourt_net_right'    : (-1, -1),
		'vcourt_center'       : (-1, -1),
		'vcourt_top_left'     : (-1, -1),
		'vcourt_top_right'    : (-1, -1),
		'plane_homography_filename' : './clip4_plane_homography.txt',
		'player_green1_court_position_filename' : './clip4_player_green1_court_position.txt',
		'player_green2_court_position_filename' : './clip4_player_green2_court_position.txt',
		'player_white1_court_position_filename' : './clip4_player_white1_court_position.txt',
		'player_white2_court_position_filename' : './clip4_player_white2_court_position.txt',
		'player_ball_court_position_filename' : './clip4_ball_court_position.txt'
	},
	'clip5' : {
		'filename'       : './beachVolleyball/beachVolleyball5.mov',
		'panorama_filename' : './beachVolleyball5_panorama.mov',
		'panorama_bg_filename' : './beachVolleyball5_panorama_bg.jpg',
		'panorama_final_filename' : "./beachVolleyball5_panorama_final.mov",
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
		},
		'player_red1_tracking_end_frame' : 999,
		'player_red1_tracking_start_frame' : 0,
		'player_red1_position_filename' : './clip5_player_red1_position.txt',
		'player_red2_tracking_end_frame' : 999,
		'player_red2_tracking_start_frame' : 0,
		'player_red2_position_filename' : './clip5_player_red2_position.txt',
		'player_white1_tracking_end_frame' : 999,
		'player_white1_tracking_start_frame' : 0,
		'player_white1_position_filename' : './clip5_player_white1_position.txt',
		'player_white2_tracking_end_frame' : 999,
		'player_white2_tracking_start_frame' : 0,
		'player_white2_position_filename' : './clip5_player_white2_position.txt',
		'player_ball_tracking_end_frame' : 999,
		'player_ball_tracking_start_frame' : 0,
		'player_ball_position_filename' : './clip5_ball_position.txt',
		'vcourt_points' : ['vcourt_bottom_left','vcourt_bottom_right','vcourt_net_left','vcourt_net_right','vcourt_center'],
		'vcourt_bottom_left'  : (-1, -1),
		'vcourt_bottom_right' : (671, 326),
		'vcourt_net_left'     : (80, 197),
		'vcourt_net_right'    : (421, 185),
		'vcourt_center'       : (253, 192),
		'vcourt_top_left'     : (-1, -1),
		'vcourt_top_right'    : (-1, -1),
		'plane_homography_filename' : './clip5_plane_homography.txt',
		'player_red1_court_position_filename' : './clip5_player_red1_court_position.txt',
		'player_red2_court_position_filename' : './clip5_player_red2_court_position.txt',
		'player_white1_court_position_filename' : './clip5_player_white1_court_position.txt',
		'player_white2_court_position_filename' : './clip5_player_white2_court_position.txt',
		'player_ball_court_position_filename' : './clip5_ball_court_position.txt'
	},
	'clip6' : {
		'filename'       : './beachVolleyball/beachVolleyball6.mov',
		'panorama_filename' : './beachVolleyball6_panorama.mov',
		'panorama_bg_filename' : './beachVolleyball6_panorama_bg.jpg',
		'panorama_final_filename' : "./beachVolleyball6_panorama_final.mov",
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
		},
		'player_red1_tracking_end_frame' : 378,
		'player_red1_tracking_start_frame' : 0,
		'player_red1_position_filename' : './clip6_player_red1_position.txt',
		'player_red2_tracking_end_frame' : 378,
		'player_red2_tracking_start_frame' : 0,
		'player_red2_position_filename' : './clip6_player_red2_position.txt',
		'player_white1_tracking_end_frame' : 378,
		'player_white1_tracking_start_frame' : 0,
		'player_white1_position_filename' : './clip6_player_white1_position.txt',
		'player_white2_tracking_end_frame' : 378,
		'player_white2_tracking_start_frame' : 0,
		'player_white2_position_filename' : './clip6_player_white2_position.txt',
		'player_ball_tracking_end_frame' : 378,
		'player_ball_tracking_start_frame' : 0,
		'player_ball_position_filename' : './clip6_ball_position.txt',
        'vcourt_points' : ['vcourt_bottom_right', 'vcourt_net_right','vcourt_center','vcourt_exp1'],
        'vcourt_bottom_left'  : (-1, -1),
        'vcourt_bottom_right' : (695, 242),
        'vcourt_net_left'     : (-1, -1),
        'vcourt_net_right'    : (333, 234),
        'vcourt_center'       : (330, 304),
        'vcourt_top_left'     : (-1, -1),
        'vcourt_top_right'    : (-1, -1),
        'vcourt_exp1' 	      : (288, 250),
		'plane_homography_filename' : './clip6_plane_homography.txt',
		'player_red1_court_position_filename' : './clip6_player_red1_court_position.txt',
		'player_red2_court_position_filename' : './clip6_player_red2_court_position.txt',
		'player_white1_court_position_filename' : './clip6_player_white1_court_position.txt',
		'player_white2_court_position_filename' : './clip6_player_white2_court_position.txt',
		'player_ball_court_position_filename' : './clip6_ball_court_position.txt'
	},
	'clip7' : {
		'filename'          : './beachVolleyball/beachVolleyball7.mov',
		'panorama_filename' : './beachVolleyball7_panorama.mov',
		'panorama_bg_filename' : './beachVolleyball7_panorama_bg.jpg',
		'panorama_final_filename' : "./beachVolleyball7_panorama_final.mov",
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
		},
		'player_red1_tracking_end_frame' : 999,
		'player_red1_tracking_start_frame' : 0,
		'player_red1_position_filename' : './clip7_player_red1_position.txt',
		'player_red2_tracking_end_frame' : 999,
		'player_red2_tracking_start_frame' : 0,
		'player_red2_position_filename' : './clip7_player_red2_position.txt',
		'player_white1_tracking_end_frame' : 999,
		'player_white1_tracking_start_frame' : 0,
		'player_white1_position_filename' : './clip7_player_white1_position.txt',
		'player_white2_tracking_end_frame' : 999,
		'player_white2_tracking_start_frame' : 0,
		'player_white2_position_filename' : './clip7_player_white2_position.txt',
		'player_ball_tracking_end_frame' : 999,
		'player_ball_tracking_start_frame' : 0,
		'player_ball_position_filename' : './clip7_ball_position.txt',
		'vcourt_points' : ['vcourt_bottom_left','vcourt_bottom_right','vcourt_net_left','vcourt_net_right','vcourt_center'],
		'vcourt_bottom_left'  : (616, 384),
		'vcourt_bottom_right' : (612, 220),
		'vcourt_net_left'     : (259, 128),
		'vcourt_net_right'    : (259, 208),
		'vcourt_center'       : (259, 278),
		'vcourt_top_left'     : (-94, 220),
		'vcourt_top_right'    : (-98, 384),
		'plane_homography_filename' : './clip7_plane_homography.txt',
		'player_red1_court_position_filename' : './clip7_player_red1_court_position.txt',
		'player_red2_court_position_filename' : './clip7_player_red2_court_position.txt',
		'player_white1_court_position_filename' : './clip7_player_white1_court_position.txt',
		'player_white2_court_position_filename' : './clip7_player_white2_court_position.txt',
		'player_ball_court_position_filename' : './clip7_ball_court_position.txt'
	}
}


PANORAMA_BLEND = {
    'PANORAMA_FRAME_STEP': 10,
    'TRANSPARENT_THRESHOLD': 10
}

def motion_tracking(filename, ROI, start=0, end=None, maxCorners=3, skip=None):
	""" 
		Acknowledge : Lucas-Kanade optical flow tracker - Code taken from http://docs.opencv.org/trunk/d7/d8b/tutorial_py_lucas_kanade.html
		with some modifications for our own use

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
			if is_cv3():
				mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
				frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
			else: 
				cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
				cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
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

def is_cv2():
	# if we are using OpenCV 2, then our cv2.__version__ will start
	# with '2.'
	return check_opencv_version("2.")
 
def is_cv3():
	# if we are using OpenCV 3.X, then our cv2.__version__ will start
	# with '3.'
	return check_opencv_version("3.")
 
def check_opencv_version(major, lib=None):
	# if the supplied library is None, import OpenCV
	if lib is None:
		import cv2 as lib
		
	# return whether or not the current OpenCV version matches the
	# major version number
	return lib.__version__.startswith(major)