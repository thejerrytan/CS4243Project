from __future__ import division

import matplotlib.pyplot as plt
import numpy
import numpy as np
import math as M
from scipy.signal import butter, lfilter, freqz

WINDOW_SIZE = 10


def plot_topdown(clip):
    player1 = np.loadtxt('./plot/%s_player_%d.txt' % (clip, 1))
    player2 = np.loadtxt('./plot/%s_player_%d.txt' % (clip, 2))
    player3 = np.loadtxt('./plot/%s_player_%d.txt' % (clip, 3))
    player4 = np.loadtxt('./plot/%s_player_%d.txt' % (clip, 4))
    ball = np.loadtxt('./plot/%s_ball.txt' % (clip))

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
        if i % 100 == 0:
            print i

        plt.clf()
        plt.xlim([-1000, 1000])
        plt.ylim([-1000, 1000])
        plt.plot((-800, 800), (-400, -400), "k-")
        plt.plot((-800, 800), (400, 400), "k-")
        plt.plot((800, 800), (-400, 400), "k-")
        plt.plot((-800, -800), (-400, 400), "k-")
        plt.plot((0, 0), (500, -500), "k-")
        plt.scatter(player1[i, 0], player1[i, 1], edgecolor='green', facecolor='green', s=9 ** 2, marker='o')
        plt.scatter(player2[i, 0], player2[i, 1], edgecolor='green', facecolor='green', s=9 ** 2, marker='o')
        plt.scatter(player3[i, 0], player3[i, 1], edgecolor='red', facecolor='red', s=9 ** 2, marker='o')
        plt.scatter(player4[i, 0], player4[i, 1], edgecolor='red', facecolor='red', s=9 ** 2, marker='o')
        # plt.scatter(ball[i,0],ball[i,1],edgecolor='blue',facecolor='blue',s=9**2,marker='o')

        plt.show()
        plt.pause(0.016)

    return


def process_motion():
    return


def detect_jumps(player):
    det_win = []
    for i in range(0, len(player)):
        if i < WINDOW_SIZE:
            det_win.append(player[i])

    return


def savitzky_golay(y, window_size, order, deriv=0, rate=1):

    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def low_pass(rawdata):
    import scipy.signal as signal
    N = 3  # Filter order
    Wn = 0.1  # Cutoff frequency
    B, A = signal.butter(N, Wn, output='ba')
    smooth_data = signal.filtfilt(B, A, rawdata)
    return smooth_data


def smooth(x):
    return low_pass(x)
    # return savitzky_golay(x, 51, 3)


def sqdistance(x1, y1, x2, y2):
    return (x1 - x2)*(x1-x2) + (y1-y2)*(y1-y2)


def smooth_position_and_count_jump(arr):
    THRESHOLD = 10
    n = len(arr)
    isJumping = False
    jumpCount = 0
    for i in range(1, n):
        x1 = arr[i-1, 0]
        y1 = arr[i-1, 1]
        x2 = arr[i, 0]
        y2 = arr[i, 1]
        if sqdistance(x1, y1, x2, y2) > THRESHOLD:
            if not isJumping:
                jumpCount += 1
                isJumping = True
            arr[i] = arr[i-1]
        else:
            isJumping = False
    return arr, jumpCount

##############################################################################
# Works on the basis that ball moves in a straight line from player to player
# and ends on the last frame of the video cut (or certain specified frame)
# keeps a record of all these 'valid' ball conditions
##############################################################################
def ball_positions(p1, p2, p3, p4, ball, threshold):
    frame_num = []

    for i in range(0, len(player)):  # tracks and stores the frame whr ball and player meets
        if ball[i] <= p1[i] + threshold or ball[i] >= p1[i] + threshold:
            frame_num.append[i]

    frame_num.append[len(player)]

    return frame_num


if __name__ == '__main__':
    plot_topdown('clip1')
