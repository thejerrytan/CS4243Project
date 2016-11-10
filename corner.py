import numpy as np
import numpy.linalg as la
import cv2
import matplotlib.pyplot as plt
import math

def MyConvolve(img, ff):
	# flip the filter
	ff = np.fliplr(ff)
	ff = np.flipud(ff)
	result = np.zeros(img.shape)
	# We do not take care of boundary pixels in this simplified implementation
	for i in range(1,img.shape[0]-1):
		for j in range(1, img.shape[1]-1):
			view = 	np.array(img[i-1:i+2,[j-1,j,j+1]])
			value = np.sum(np.multiply(view, ff))
			result[i,j] = value
	return result

def show_corner(image, x, y):
	plt.figure()
	plt.imshow(image, cmap='gray')
	plt.hold(True)
	plt.scatter(x,y,color='blue',s=10)
	plt.show()

def gauss_kernels(size,sigma=1.0):
	## returns a 2d gaussian kernel
	if size<3:
		size = 3
	m = size/2
	x, y = np.mgrid[-m:m+1, -m:m+1]
	kernel = np.exp(-(x*x + y*y)/(2*sigma*sigma))
	kernel_sum = kernel.sum()
	if not kernel_sum==0:
		kernel = kernel/kernel_sum
	return kernel

def generate_scatter(c):
	# Top border
	x = np.array([c[0]-4, c[0]-3, c[0]-2, c[0]-1, c[0], c[0]+1, c[0]+2, c[0]+3, c[0]+4])
	y = np.array([c[1]-4] * 9)
	# Bottom border
	x = np.hstack((x, x))
	y = np.hstack(([c[1]+4] * 9, y))
	# Left border
	x = np.hstack(([c[0]-4] * 7, x))
	y = np.hstack(([c[1]-3, c[1]-2, c[1]-1, c[1], c[1]+1, c[1]+2, c[1]+3], y))
	# Right border
	x = np.hstack(([c[0]+4] * 7, x))
	y = np.hstack(([c[1]-3, c[1]-2, c[1]-1, c[1], c[1]+1, c[1]+2, c[1]+3], y))
	return (y, x)

def harris_corner(image, thres=0.9, step=10):
	# Edge detection
	sorbel_h = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
	sorbel_v = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
	
	gx = MyConvolve(image, sorbel_h)
	gy = MyConvolve(image, sorbel_v)

	I_xx = np.multiply(gx, gx)
	I_xy = np.multiply(gx, gy)
	I_yy = np.multiply(gy, gy)

	W_xx = MyConvolve(I_xx, gauss_kernels(3))
	W_xy = MyConvolve(I_xy, gauss_kernels(3))
	W_yy = MyConvolve(I_yy, gauss_kernels(3))

	k = 0.06
	values = []
	for i in range(0, W_xx.shape[0], step):
		for j in range(0, W_xx.shape[1], step):
			W = np.matrix([
						   [W_xx[i,j], W_xy[i,j]],
						   [W_xy[i,j], W_yy[i,j]]
						  ])
			detW = la.det(W)
			traceW = np.trace(W)
			response = detW - (k * traceW * traceW)
			values.append((i, j, response))

	# Find good corners
	max_response = max(values, key=lambda x: x[2])
	# print max_response
	# print values
	good_corners = filter(lambda x: abs(x[2]) >= thres * max_response[2], values)
	# print good_corners
	x = np.array([])
	y = np.array([])
	for c in good_corners:
		x = np.hstack((x, generate_scatter(c)[0]))
		y = np.hstack((y, generate_scatter(c)[1]))
	show_corner(image, x, y)
	return (good_corners, x, y)

def saveFile(img, filename, x, y):
	for (i,j) in zip(x, y):
		try:
			# Bounding windows may exceed img boundaries
			img[j,i,:] = (255,0,0)
		except IndexError as e:
			pass
	cv2.imwrite(filename, img)

def test():
	test1   = cv2.imread('./test1.jpg', cv2.IMREAD_GRAYSCALE)
	test2   = cv2.imread('./test2.jpg', cv2.IMREAD_GRAYSCALE)
	test3   = cv2.imread('./test3.jpg', cv2.IMREAD_GRAYSCALE)
	checker = cv2.imread('./checker.jpg', cv2.IMREAD_GRAYSCALE)
	flower  = cv2.imread('./flower.jpg', cv2.IMREAD_GRAYSCALE)

	# THRESHOLD 0.1, step=10
	(test1_ten_corners, test1_ten_window_x, test1_ten_window_y)       = harris_corner(test1, 0.1, 10)
	(test2_ten_corners, test2_ten_window_x, test2_ten_window_y)       = harris_corner(test2, 0.1, 10)
	(test3_ten_corners, test3_ten_window_x, test3_ten_window_y)       = harris_corner(test3, 0.1, 10)
	(checker_ten_corners, checker_ten_window_x, checker_ten_window_y) = harris_corner(checker, 0.1, 10)
	(flower_ten_corners, flower_ten_window_x, flower_ten_window_y)    = harris_corner(flower, 0.1, 10)
	
	(test1_one_corners, test1_one_window_x, test1_one_window_y)       = harris_corner(test1, 0.1, 1)
	(test2_one_corners, test2_one_window_x, test2_one_window_y)       = harris_corner(test2, 0.1, 1)
	(test3_one_corners, test3_one_window_x, test3_one_window_y)       = harris_corner(test3, 0.1, 1)
	(checker_one_corners, checker_one_window_x, checker_one_window_y) = harris_corner(checker, 0.1, 1)
	(flower_one_corners, flower_one_window_x, flower_one_window_y)    = harris_corner(flower, 0.1, 1)
	
	# Save results
	test1_c   = cv2.imread('./test1.jpg')
	test2_c   = cv2.imread('./test2.jpg')
	test3_c   = cv2.imread('./test3.jpg')
	checker_c = cv2.imread('./checker.jpg')
	flower_c  = cv2.imread('./flower.jpg')

	saveFile(test1_c, './test1_10_corner.jpg', test1_ten_window_x, test1_ten_window_y)
	saveFile(test2_c, './test2_10_corner.jpg', test2_ten_window_x, test2_ten_window_y)
	saveFile(test3_c, './test3_10_corner.jpg', test3_ten_window_x, test3_ten_window_y)
	saveFile(flower_c, './flower_10_corner.jpg', flower_ten_window_x, flower_ten_window_y)
	saveFile(checker_c, './checker_10_corner.jpg', checker_ten_window_x, checker_ten_window_y)

	saveFile(test1_c, './test1_1_corner.jpg', test1_one_window_x, test1_one_window_y)
	saveFile(test2_c, './test2_1_corner.jpg', test2_one_window_x, test2_one_window_y)
	saveFile(test3_c, './test3_1_corner.jpg', test3_one_window_x, test3_one_window_y)
	saveFile(flower_c, './flower_1_corner.jpg', flower_one_window_x, flower_one_window_y)
	saveFile(checker_c, './checker_1_corner.jpg', checker_one_window_x, checker_one_window_y)

if __name__ == "__main__":
	test()