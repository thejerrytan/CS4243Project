from __future__ import division

import cv2
import matplotlib.pyplot as plt
import numpy as np

WINDOW_SIZE = 10


def plot_topdown(clip):
    cap = cv2.VideoCapture("./beachVolleyball/beachVolleyball%d.mov" % clip)
    player1 = np.loadtxt('./clip%d_player_green1_position.txt' % (clip))
    player2 = np.loadtxt('./clip%d_player_green2_position.txt' % (clip))
    player3 = np.loadtxt('./clip%d_player_white1_position.txt' % (clip))
    player4 = np.loadtxt('./clip%d_player_white2_position.txt' % (clip))
    ball = np.loadtxt('./clip%d_ball_position.txt' % (clip))
    interpolate_ball(ball)

    frames = len(player1)

    # player1[0] = smooth(player1[:, 0])
    # player1[1] = smooth(player1[:, 1])

    # print smooth(np.array([0, 1, 2, 3, 15, 3, 5, 6, 7, 8, 10, 8, 10, 11, 12]))

    _, jump1 = smooth_position_and_count_jump(player1)
    _, jump2 = smooth_position_and_count_jump(player2)
    _, jump3 = smooth_position_and_count_jump(player3)
    _, jump4 = smooth_position_and_count_jump(player4)

    print "Jump Count 1: ", jump1
    print "Jump Count 2: ", jump2
    print "Jump Count 3: ", jump3
    print "Jump Count 4: ", jump4

    plt.figure()
    plt.ion()

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
        plt.scatter(player1[i, 1], player1[i, 0], edgecolor='green', facecolor='green', s=9 ** 2, marker='o')
        plt.scatter(player2[i, 1], player2[i, 0], edgecolor='green', facecolor='green', s=9 ** 2, marker='o')
        plt.scatter(player3[i, 1], player3[i, 0], edgecolor='red', facecolor='red', s=9 ** 2, marker='o')
        plt.scatter(player4[i, 1], player4[i, 0], edgecolor='red', facecolor='red', s=9 ** 2, marker='o')
        plt.scatter(ball[i, 1], ball[i, 0], edgecolor='blue', facecolor='blue', s=9 ** 2, marker='o')

        plt.show()
        plt.pause(0.0005)

        _, frame = cap.read()
        cv2.imshow("original:", frame)

    return


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
    plot_topdown(3)
