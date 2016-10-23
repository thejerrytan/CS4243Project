import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.linalg as la

OFFSET_U  = 0
OFFSET_V  = 0
BETA_U    = 1
BETA_V    = 1
K_U       = 1
K_V       = 1
FOCAL_LEN = 1

pts      = np.zeros([11,3])
pts[0,:] = [-1,-1,-1]
pts[1,:] = [1, -1, -1]
pts[2,:] = [1,1,-1]
pts[3,:] = [-1,1,-1]
pts[4,:] = [-1,-1,1]
pts[5,:] = [1,-1,1]
pts[6,:] = [1,1,1]
pts[7,:] = [-1,1,1]
pts[8,:] = [-0.5,-0.5,-1]
pts[9,:] = [0.5,-0.5,-1]
pts[10,:] = [0,0.5,-1]

def quatconjugate(q1):
	"""returns q1*"""
	return np.hstack((q1[0], -1 * q1[1:]))

def quatmult(q1, q2):
	"""q1 is (1,4) np.array, q2 is (1,4) np.array, returns a quaternion representation of a point"""
	v1 = q1[1:]
	v2 = q2[1:]
	r0 = q1[0] * q2[0] - np.dot(v1, v2)
	r1 = np.cross(v1,v2) + q1[0] * v2 + q2[0] * v1
	return np.hstack((r0, r1))

def pt_to_quat(r):
	""" Given a vector r convert to quaternion"""
	return np.hstack((0, r))

def quat2pt(q):
	return q[1:]

def rot(angle, axis):
	""" Given angle in degrees, and axis of rotation, return a quaternion representing rotation of theta about axis"""
	rot = np.array([
		math.cos(math.pi * angle/180.0 * 0.5), 
		math.sin(math.pi * angle/180.0 * 0.5) * axis[0], 
		math.sin(math.pi * angle/180.0 * 0.5) * axis[1], 
		math.sin(math.pi * angle/180.0 * 0.5) * axis[2]])
	return rot

def quat2rot(q):
	""" Given quaterion representing a rotation, return a rotation matrix"""
	r00 = q[0]*q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3]
	r01 = 2*(q[1]*q[2] - q[0]*q[3])
	r02 = 2*(q[1]*q[3] + q[0]*q[2])
	r10 = 2*(q[1]*q[2] + q[0]*q[3])
	r11 = q[0]*q[0] + q[2]*q[2] - q[1]*q[1] - q[3]*q[3]
	r12 = 2*(q[2]*q[3] - q[0]*q[1])
	r20 = 2*(q[1]*q[3] - q[0]*q[2])
	r21 = 2*(q[2]*q[3] + q[0]*q[1])
	r22 = q[0]*q[0] + q[3]*q[3] - q[1]*q[1] - q[2]*q[2]

	rot = np.matrix([
		[r00, r01, r02],
		[r10, r11, r12],
		[r20, r21, r22]
		])
	return rot

def plot(fig_num, p1, p2, p3, p4):
	fig = plt.figure(fig_num)
	if fig_num == 1:
		fig.suptitle("Orthographic")
	else:
		fig.suptitle("Perspective")
	sp1 = fig.add_subplot(221)
	sp2 = fig.add_subplot(222)
	sp3 = fig.add_subplot(223)
	sp4 = fig.add_subplot(224)
	sp1.set_xlim(-2,2)
	sp1.set_ylim(-2,2)
	sp2.set_xlim(-2,2)
	sp2.set_ylim(-2,2)
	sp3.set_xlim(-2,2)
	sp3.set_ylim(-2,2)
	sp4.set_xlim(-2,2)
	sp4.set_ylim(-2,2)
	sp1.set_title("First frame")
	sp2.set_title("Second frame")
	sp3.set_title("Third frame")
	sp4.set_title("Fourth frame")
	sp1.scatter(p1[0], p1[1], marker='x')
	sp2.scatter(p2[0], p2[1], marker='x')
	sp3.scatter(p3[0], p3[1], marker='x')
	sp4.scatter(p4[0], p4[1], marker='x')
	plt.show()

def ortho_proj(pos, orient):
	""" 
	Given pos - position of camera wrt world origin, orient - orientation of camera wrt world origin,
	return orthographic projection of pts onto image plane.
	Define (0,0) of image to be center
	"""
	x = []
	y = []
	for pt in pts:
		try:
			u = np.dot((pt - pos), np.array(orient[0,:]).flatten()) * BETA_U + OFFSET_U
			v = np.dot((pt - pos), np.array(orient[1,:]).flatten()) * BETA_V + OFFSET_V
			x.append(u)
			y.append(v)
		except Exception as e:
			print e
	return (x,y)

def per_proj(pos, orient):
	""" Perspective projection"""
	x = []
	y = []
	for pt in pts:
		try:
			num_x = np.dot((pt - pos), np.array(orient[0,:]).flatten())
			num_y = np.dot((pt - pos), np.array(orient[1,:]).flatten())
			denom = np.dot((pt - pos), np.array(orient[2,:]).flatten())
			u = (FOCAL_LEN * num_x / denom) * BETA_U + OFFSET_U
			v = (FOCAL_LEN * num_y / denom) * BETA_V + OFFSET_V
			x.append(u)
			y.append(v)
		except Exception as e:
			print e
	return (x,y)

def calc_homography(pts, cam):
	"""
		Returns H, given a set of points on the plane [up, vp], and points on the camera image plane cam[uc, vc],
		calculate the homography matrix H between them.
		At least 5 points must be given to be able to solve for H
	"""
	# 3: Homography - using pts and taking u_c, v_c to be the coordinates of the image under perspective projection
	A = np.zeros((1,9))
	for i in range(0, pts.shape[0]):
		A = np.vstack((A,
			np.array([
				[pts[i,0], pts[i,1], 1, 0, 0, 0, -1*cam[i,0]*pts[i,0], -1*cam[i,0]*pts[i,1], -1*cam[i,0]],
				[0, 0, 0, pts[i,0], pts[i,1], 1, -1*cam[i,1]*pts[i,0], -1*cam[i,1]*pts[0,1], -1*cam[i,1]]
			]))
		)
	U, S, V = la.svd(np.matrix(A))
	print(S[-1])
	H = np.reshape(V[-1,:], (3,3))
	# Normalize H
	H = np.divide(H, H[2,2])
	return H

def test():
	q1 = np.array([0,1,1,1])
	q2 = np.array([1,0.5,0.5,0.5])
	# print(quatmult(q1,q2))
	# 1.2 Getting camera position
	rotation_axis = np.array([0,1,0])
	rot30 = rot(-30, rotation_axis)
	rot60 = rot(-60, rotation_axis)
	rot90 = rot(-90, rotation_axis)
	rot360 = rot(-360 , rotation_axis)
	r  = np.array([0,0,-5])
	p  = pt_to_quat(r)
	f2 = quatmult(quatmult(rot30, p), quatconjugate(rot30))
	f3 = quatmult(quatmult(rot60, p), quatconjugate(rot60))
	f4 = quatmult(quatmult(rot90, p), quatconjugate(rot90))
	# print(r)
	# print(f2)
	# print(f3)
	# print(f4)
	# 1.3 Getting camera orientation
	quatmat_1 = np.eye(3)
	quatmat_2 = quat2rot(rot(30, rotation_axis))
	quatmat_3 = quat2rot(rot(60, rotation_axis))
	quatmat_4 = quat2rot(rot(90, rotation_axis))
	# print(quatmat_1)
	# print(quatmat_2)
	# print(quatmat_3)
	# print(quatmat_4)

	# 2.1 Projection using orthographic
	t2 = quat2pt(f2)
	t3 = quat2pt(f3)
	t4 = quat2pt(f4)
	p1 = ortho_proj(r, quatmat_1)
	p2 = ortho_proj(t2, quatmat_2)
	p3 = ortho_proj(t3, quatmat_3)
	p4 = ortho_proj(t4, quatmat_4)
	plot(1, p1, p2, p3, p4)

	# 2.2 Projection using perspective
	(u1,v1) = per_proj(r, quatmat_1)
	(u2,v2) = per_proj(t2, quatmat_2)
	(u3,v3) = per_proj(t3, quatmat_3)
	(u4,v4) = per_proj(t4, quatmat_4)
	plot(2, (u1,v1), (u2,v2), (u3,v3), (u4,v4))

if __name__ == "__main__":
	test()
