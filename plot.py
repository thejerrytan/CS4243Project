from __future__ import division

import glob
import math
import os
import subprocess

import cv2
import matplotlib.pyplot as plt
import numpy as np

from roi import *
version_flag = 2
if is_cv2():
    version_flag = 2
    import cv2.cv as cv
elif is_cv3():
    version_flag = 3

WINDOW_SIZE = 10


def plot_topdown(clip):
    cap = cv2.VideoCapture("./beachVolleyball/beachVolleyball%d.mov" % clip)
    color1 = "green"
    color2 = "white"
    if clip >= 5:
        color1 = "red"

    player1 = np.loadtxt('./clip%d_player_%s1_court_position.txt' % (clip, color1))
    player2 = np.loadtxt('./clip%d_player_%s2_court_position.txt' % (clip, color1))
    player3 = np.loadtxt('./clip%d_player_%s1_court_position.txt' % (clip, color2))
    player4 = np.loadtxt('./clip%d_player_%s2_court_position.txt' % (clip, color2))
    ball = np.loadtxt('./clip%d_ball_court_position.txt' % (clip))
    interpolate_ball(ball)

    frames = len(player1)

    # player1[0] = smooth(player1[:, 0])
    # player1[1] = smooth(player1[:, 1])

    # print smooth(np.array([0, 1, 2, 3, 15, 3, 5, 6, 7, 8, 10, 8, 10, 11, 12]))

    preprocess_player(player1)
    preprocess_player(player2)
    preprocess_player(player3)
    preprocess_player(player4)

    _, jump1 = smooth_position_and_count_jump(player1)
    _, jump2 = smooth_position_and_count_jump(player2)
    _, jump3 = smooth_position_and_count_jump(player3)
    _, jump4 = smooth_position_and_count_jump(player4)

    distance_1 = calculate_total_distance(player1)
    distance_2 = calculate_total_distance(player2)
    distance_3 = calculate_total_distance(player3)
    distance_4 = calculate_total_distance(player4)

    print "Jump 1: ", jump1, "Distance 1: ", distance_1
    print "Jump 2: ", jump2, "Distance 2: ", distance_2
    print "Jump 3: ", jump3, "Distance 3: ", distance_3
    print "Jump 4: ", jump4, "Distance 4: ", distance_4

    with open("./clip%d_stats.txt" % clip, "w") as f:
        f.write('Player %s 1\n' % color1)
        f.write('\tJumps: %d\n' % jump1)
        f.write('\tDistance: %d\n' % distance_1)

        f.write('Player %s 2\n' % color1)
        f.write('\tJumps: %d\n' % jump2)
        f.write('\tDistance: %d\n' % distance_2)

        f.write('Player %s 1\n' % color2)
        f.write('\tJumps: %d\n' % jump3)
        f.write('\tDistance: %d\n' % distance_3)

        f.write('Player %s 2\n' % color2)
        f.write('\tJumps: %d\n' % jump3)
        f.write('\tDistance: %d\n' % distance_3)

    plt.figure()
    plt.ion()

    if not os.path.exists("./topdown/clip%d" % clip):
        os.makedirs("./topdown/clip%d" % clip)
    
    if version_flag == 2:
        fourcc = cv.CV_FOURCC('m', 'p', '4', 'v')
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None

    for i in range(0, frames):
        if i % 10 == 0:
            print i

        plt.clf()
        plt.xlim([-1000, 1000])
        plt.ylim([-1000, 1000])
        plt.plot((-800, 800), (-400, -400), "k-")
        plt.plot((-800, 800), (400, 400), "k-")
        plt.plot((800, 800), (-400, 400), "k-")
        plt.plot((-800, -800), (-400, 400), "k-")
        plt.plot((0, 0), (500, -500), "k-")
        plt.scatter(player1[i, 1], player1[i, 0], edgecolor=color1, facecolor=color1, s=9 ** 2, marker='o')
        plt.scatter(player2[i, 1], player2[i, 0], edgecolor=color1, facecolor=color1, s=9 ** 2, marker='o')
        plt.scatter(player3[i, 1], player3[i, 0], edgecolor='yellow', facecolor='yellow', s=9 ** 2, marker='o')
        plt.scatter(player4[i, 1], player4[i, 0], edgecolor='yellow', facecolor='yellow', s=9 ** 2, marker='o')
        plt.scatter(ball[i, 1], ball[i, 0], edgecolor='blue', facecolor='blue', s=9 ** 2, marker='o')

        plt.savefig("./topdown/clip%d/frame.png" % (clip))
        frame = cv2.imread("./topdown/clip%d/frame.png" % (clip), cv2.IMREAD_COLOR)

        if writer is None:
            writer = cv2.VideoWriter("./topdown/clip%d.mov" % clip, fourcc, 60, (frame.shape[1], frame.shape[0]), True)

        writer.write(frame)
        plt.show()
        plt.pause(0.0005)

        _, frame = cap.read()
        cv2.imshow("original:", frame)

    writer.release()


def preprocess_player(arr):
    for i in range(1, len(arr)):
        if arr[0, 1] < 0 and arr[i, 1] > 0:
            arr[i, 1] = 0
        if arr[0, 1] > 0 and arr[i, 1] < 0:
            arr[i, 1] = 0


def interpolate_ball(arr):
    last_value = arr[0]
    last_change = 0
    for i in range(1, len(arr)):
        cur_value = arr[i]

        if cur_value[0] != last_value[0] or cur_value[1] != last_value[1]:
            # change position
            change = [(cur_value[0] - last_value[0]) / (i - last_change),
                      (cur_value[1] - last_value[1]) / (i - last_change)]

            for k in range(last_change + 1, i):
                arr[k] = [arr[k - 1][0] + change[0], arr[k - 1][1] + change[1]]

            last_change = i
            last_value = cur_value

    return arr


def sqdistance(x1, y1, x2, y2):
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)


def distance(x1, y1, x2, y2):
    return math.sqrt(sqdistance(x1, y1, x2, y2))


def calculate_total_distance(arr):
    total = 0
    for i in range(1, len(arr)):
        pt1 = arr[i - 1]
        pt2 = arr[i]
        total += distance(pt2[1], pt2[0], pt1[1], pt1[0])

    return total


def smooth_position_and_count_jump(arr):
    THRESHOLD = 10000
    ON_AIR_THRESHOLD = 60 * 1
    frameSinceJump = 0
    n = len(arr)
    isJumping = False
    jumpCount = 0
    for i in range(1, n):
        x1 = arr[i - 1, 0]
        y1 = arr[i - 1, 1]
        x2 = arr[i, 0]
        y2 = arr[i, 1]
        if sqdistance(x1, y1, x2, y2) > THRESHOLD and frameSinceJump < ON_AIR_THRESHOLD:
            if not isJumping:
                jumpCount += 1
                isJumping = True
            arr[i] = arr[i - 1]
            frameSinceJump += 1
        else:
            isJumping = False
            frameSinceJump = 0
    return arr, jumpCount


if __name__ == '__main__':
    plot_topdown(7)
