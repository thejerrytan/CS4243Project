import numpy as np
import cv2
import cv2.cv as cv
from roi import *
from matplotlib import pyplot as plt
# try:
#     import cv2.cv as cv
# except ImportError as e:
#     import cv2 as cv
from matplotlib import pyplot as plt

def findGoodMatches(img1, img2):
    # Initiate SIFT detector
    # sift = cv2.SIFT()
    #
    # # find the keypoints and descriptors with SIFT
    # kp1, des1 = sift.detectAndCompute(img1, None)
    # kp2, des2 = sift.detectAndCompute(img2, None)
    #
    # FLANN_INDEX_KDTREE = 0
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # search_params = dict(checks=50)
    #
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    #
    # matches = flann.knnMatch(des1, des2, k=2)
    #
    # # store all the good matches as per Lowe's ratio test.
    # good = []
    # for m , n in matches:
    #     if m.distance < 0.7 * n.distance:
    #         good.append(m)
    #
    # src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    # dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    src_pts = np.array([[204.0, 290.0], [438.0, 138.0], [41.0, 146.0], [294.0, 84.0]])
    dst_pts = np.array([[112.0, 296.0], [586.0, 245.0], [153.0, 139.0], [436.0, 129.0]])
    return src_pts, dst_pts


def cal_homography(img1, img2):
    srcPts, dstPts = findGoodMatches(img1, img2)
    H, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5.0)
    return H
    # img3 = np.zeros([img1.shape[0] + img2.shape[0], img1.shape[1]])
    #
    # for i in range (img1.shape[0]):
    #     for j in range (img1.shape[1]):
    #         img3[i, j] = img1[i, j]
    #         img3[i + img1.shape[0], j] = img2[i, j]
    # cv2.line(img3, (0, 0), (511, 511), (255, 0, 0), 5)

    # for i in range(len(srcPts)):
    #     srcPt = srcPts[i][0]
    #     dstPt = dstPts[i][0]
    #     dstPt[1] += img1.shape[0]
    #     print srcPt
    #     srcPt = tuple(srcPt)
    #     print srcPt
    #     dstPt = tuple(dstPt)
    #     print dstPt
    #     cv2.line(img3, srcPt, dstPt, (255,0,0), 1)


    # cv2.imwrite("img_result.jpg", img3)

    # return H

def constructPanorama(clipFileName):
    cap = cv2.VideoCapture(clipFileName)
    fw = int(cap.get(cv2.cv.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.cv.CAP_PROP_FPS))
    fc = int(cap.get(cv2.cv.CAP_PROP_FRAME_COUNT))
    image = np.zeros([fw, fh, 3])

    img1 = cap.read()[1][:, :, :]
    for i in range (600):
        img2 = cap.read()[1][:, :, :]

    cv2.imwrite("img1.jpg", img1)
    cv2.imwrite("img2.jpg", img2)

    H = cal_homography(img1, img2)
    img3 = cv2.warpPerspective(img1, H, (1000, 1000))
    cv2.imwrite("warped_img1.jpg", img3)

    # combine warped img1 and img2
    result = np.zeros([1000, 1000, 3])
    for i in range(1000):
        for j in range(1000):
            img2_pt = img3[i, j, :]
            if i < img2.shape[0] and j < img2.shape[1]:
                img2_pt = img2[i, j, :]
            result[i, j, :] = (img3[i, j, :] + img2_pt[:]) / 2

    cv2.imwrite("img_result.jpg", result)
    return image


def blendFrames(clipFileName):
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
                    if frame[h, w, 0] + frame[h, w, 1] + frame[h, w, 2] > PANORAMA_BLEND['TRANSPARENT_THRESHOLD']:
                        count[h, w] += 1
                        image[h, w, :] += frame[h, w, :]

    for h in range(fh):
        for w in range(fw):
            if (count[h, w] > 0):
                for i in range (3):
                    image[h, w, i] = float(image[h, w, i]) / count[h, w]

    return image


# CLIP1 = './beachVolleyball/beachVolleyball1.mov'

# constructPanorama(CLIP1)
image = blendFrames('./beachVolleyball/beachVolleyball6_panorama.mov')
cv2.imwrite("img_6.jpg", image)

print "done"