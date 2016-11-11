from __future__ import division
import numpy as np
import math
import numpy.linalg as la
import cv2
import matplotlib.pyplot as plt

WINDOW_SIZE = 10

def plot_topdown(clip):
	player1 = np.loadtxt('./plot/%s_player_%d.txt' % (clip,1))
	player2 = np.loadtxt('./plot/%s_player_%d.txt' % (clip,2))
	player3 = np.loadtxt('./plot/%s_player_%d.txt' % (clip,3))
	player4 = np.loadtxt('./plot/%s_player_%d.txt' % (clip,4))
	ball = np.loadtxt('./plot/%s_ball.txt' % (clip))

	frames = len(player1)

	plt.figure()
	plt.ion()

	
	for i in range(0, frames):
		print i
		plt.clf()
		plt.xlim([-1000,1000])
		plt.ylim([-1000, 1000])
		plt.plot((-800,800),(-400,-400),"k-")
		plt.plot((-800,800),(400,400),"k-")
		plt.plot((800,800),(-400,400),"k-")
		plt.plot((-800,-800),(-400,400),"k-")
		plt.plot((0,0),(500,-500),"k-")
		plt.scatter(player1[i,0],player1[i,1],edgecolor='green',facecolor='green',s=9**2,marker='o')
		plt.scatter(player2[i,0],player2[i,1],edgecolor='green',facecolor='green',s=9**2,marker='o')
		plt.scatter(player3[i,0],player3[i,1],edgecolor='red',facecolor='red',s=9**2,marker='o')
		plt.scatter(player4[i,0],player4[i,1],edgecolor='red',facecolor='red',s=9**2,marker='o')
		# plt.scatter(ball[i,0],ball[i,1],edgecolor='blue',facecolor='blue',s=9**2,marker='o')

		plt.show()
		plt.pause(0.016)
	return

def process_motion():

	return

def detect_jumps(player):
	det_win = []
	for i in range(0,len(player)):
		if i < WINDOW_SIZE :
			det_win.append(player[i])

	return


##############################################################################
# Works on the basis that ball moves in a straight line from player to player
# and ends on the last frame of the video cut (or certain specified frame)
# keeps a record of all these 'valid' ball conditions
##############################################################################
def ball_positions(p1,p2,p3,p4,ball,threshold):

	frame_num = []

	for i in range(0,len(player)): # tracks and stores the frame whr ball and player meets
		if ball[i] <= p1[i]+threshold or ball[i] >= p1[i]+threshold:
			frame_num.append[i]

	frame_num.append[len(player)]

	return frame_num

if __name__ == '__main__' :
	plot_topdown('clip1')

