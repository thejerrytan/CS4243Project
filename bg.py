import numpy as np
import cv2
import cv2.cv as cv
from roi import *
from matplotlib import pyplot as plt

def isInBound(x, y, w, h):
    return 0 <= x < w and 0 <= y < h

def isValidPoint(h, w, fh, fw, img):
    rec_size = 5

    # top left
    th = h - rec_size
    tw = w - rec_size

    if isInBound(th, tw, fh, fw):
        if (img[th, tw, 0] + img[th, tw, 1] + img[th, tw, 2] < PANORAMA_BLEND['TRANSPARENT_THRESHOLD']):
            return False
    # bottom left
    th = h + rec_size
    tw = w - rec_size
    if isInBound(th, tw, fh, fw):
        if (img[th, tw, 0] + img[th, tw, 1] + img[th, tw, 2] < PANORAMA_BLEND['TRANSPARENT_THRESHOLD']):
            return False

    # top right
    th = h - rec_size
    tw = w + rec_size
    if isInBound(th, tw, fh, fw):
        if (img[th, tw, 0] + img[th, tw, 1] + img[th, tw, 2] < PANORAMA_BLEND['TRANSPARENT_THRESHOLD']):
            return False

    # bottom right
    th = h + rec_size
    tw = w + rec_size
    if isInBound(th, tw, fh, fw):
        if (img[th, tw, 0] + img[th, tw, 1] + img[th, tw, 2] < PANORAMA_BLEND['TRANSPARENT_THRESHOLD']):
            return False

    # the point itself

    th = h
    tw = w
    if isInBound(th, tw, fh, fw):
        if (img[th, tw, 0] + img[th, tw, 1] + img[th, tw, 2] < PANORAMA_BLEND['TRANSPARENT_THRESHOLD']):
            return False

    # All passed, return true
    return True

def blendFrames(clip):
    clipFileName = PANORAMA_ROI[clip]['panorama_filename']
    cap = cv2.VideoCapture(clipFileName)
    fw = int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CV_CAP_PROP_FPS))
    fc = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))

    image = np.zeros([fh, fw, 3], float)
    count = np.zeros([fh, fw])

    print "frame count = ", fc

    for i in range(fc):
        ret, frame = cap.read()
        if i % PANORAMA_BLEND['PANORAMA_FRAME_STEP'] == 0:
            print i
            for h in range (fh):
                for w in range (fw):
                    # if frame[h, w, 0] + frame[h, w, 1] + frame[h, w, 2] > PANORAMA_BLEND['TRANSPARENT_THRESHOLD']:
                    if isValidPoint(h, w, fh, fw, frame):
                        count[h, w] += 1
                        image[h, w, :] += frame[h, w, :]

    for h in range(fh):
        for w in range(fw):
            if (count[h, w] > 0):
                for i in range (3):
                    image[h, w, i] = float(image[h, w, i]) / count[h, w]

    cv2.imwrite(PANORAMA_ROI[clip]["panorama_bg_filename"], image)
    return image