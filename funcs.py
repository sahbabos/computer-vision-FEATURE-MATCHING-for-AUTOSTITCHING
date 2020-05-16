import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random as rand
import math
import os
from scipy import ndimage
from scipy import misc
from PIL import Image
from skimage.draw import polygon
from skimage.transform import *
from skimage.feature import corner_harris, peak_local_max
from skimage.draw import polygon_perimeter as pp
from skimage.filters import gaussian
from scipy.interpolate import *
from skimage.color import rgb2gray
from scipy.ndimage.morphology import  distance_transform_edt

def load_image(filepath):
    img = mpimg.imread(filepath)
    plt.subplot()
    plt.imshow(img)
    return img/255.0
    
def getpoints(image):
	plt.imshow(image)
	points = plt.ginput(4, timeout = -1, mouse_add = 1)
	plt.close() 
	return np.array(points)


def computeH(point_1,point_2):
	string = []
	vector = []
	for i in range(4):
		x = point_1[i][0]
		y = point_1[i][1]
		x_p = point_2[i][0]
		y_p = point_2[i][1]
		string.append(['{}'.format(-x), '{}'.format(-y), '-1', '0', '0', '0', '{}'.format(x*x_p), '{}'.format(y*x_p)])
		string.append(['0', '0', '0', '{}'.format(-x), '{}'.format(-y), '-1', '{}'.format(x*y_p), '{}'.format(y*y_p)])

		vector.append(('{}'.format(-x_p), '{}'.format(-y_p)))

	A = np.array(string).astype(np.float)
	b = np.reshape(np.array(vector).astype(np.float),(8,1))
	matrix = np.linalg.lstsq(A, b)[0]
	# print (np.reshape(np.vstack((matrix, np.array(1.0))), (3,3)))
	H = np.reshape(np.vstack((matrix, np.array(1.0))), (3,3))

	# xmax, xmin = H.max(), H.min()
	# H = (H - xmin)/(xmax - xmin)
	return H
## need to rewize
def Warp(image, H, output_shape = (2000,2666,3)):
	# corners = np.array([[0, 0], [0, image.shape[1]], [image.shape[0], image.shape[1]]
	# 	, [image.shape[1], 0]])
	corners =  np.array([[0, 0], [image.shape[1] - 1, 0 ], [image.shape[1] - 1, image.shape[0] - 1], [0, image.shape[0] - 1]])
	
	
	vec1 = np.vstack((corners.T, np.ones(len(corners))))
	tpoints = np.dot(H, vec1)
	for i in range(tpoints.shape[1]):
		corner = tpoints[:, i]
		normCorner = np.dot(corner, 1/(corner[2]))
		tpoints[:, i] = normCorner
	tpoints = tpoints[:-1, :]

	x = tpoints[:,0].T
	y = tpoints[:,1].T
	row = x - int(np.floor(min(x)))
	col = y - int(np.floor(min(y)))
	rr, cc = polygon(row, col)
	rowin, colin = rr + int(np.floor(min(x))), cc + int(np.floor(min(y)))
	prow, pcol = pp(x, y)
	allx = np.append(rowin, prow)
	ally = np.append(colin, pcol)
	tPs = np.vstack((allx, ally)).T
	opoints = tp(np.linalg.inv(H), tPs)
	def interpol(func, x, y):
		return dfitpack.bispeu(func.tck[0], func.tck[1], func.tck[2], func.tck[3], func.tck[4], x, y)[0]
	
	Rchan = interp2d(range(image.shape[1]), range(image.shape[0]), image[:, :, 0])
	channle1 = interpol(Rchan, opoints[:,0], opoints[:, 1])
	Gchan = interp2d(range(image.shape[1]), range(image.shape[0]), image[:, :, 1])
	channle2 = interpol(Gchan, opoints[:,0], opoints[:, 1])
	Bchan = interp2d(range(image.shape[1]), range(image.shape[0]), image[:, :, 2])
	channle3 = interpol(Bchan, opoints[:,0], opoints[:, 1])
	
	
	mw = int(np.floor(min(tpoints[:, 0])))
	mh = int(np.floor(min(tpoints[:, 1])))

	# Mw = int(np.ceil(max(tpoints[:, 0])))
	# Mh = int(np.ceil(max(tpoints[:, 1])))
	# dimw = Mw - mw + 1
	# dimh = Mh - mh + 1

	# raw_image = np.zeros((dimh, dimw, 3))
	raw_image = np.zeros(output_shape)
	raw_image[tPs[:, 1] - mh, tPs[:, 0] - mw, 0] = channle1
	raw_image[tPs[:, 1] - mh, tPs[:, 0] - mw, 1] = channle2
	raw_image[tPs[:, 1] - mh, tPs[:, 0] - mw, 2] = channle3

	return np.clip(raw_image, 0, 1)


def tp(Homog, vec):
	vec1 = np.vstack((vec.T, np.ones(len(vec))))
	tpoints = np.dot(Homog, vec1)
	for i in range(tpoints.shape[1]):
		corner = tpoints[:, i]
		normCorner = np.dot(corner, 1/(corner[2]))
		tpoints[:, i] = normCorner

	return tpoints[:-1, :].T


def blender(image1, image2, image3):
	alphamaskzero = np.zeros(image1.shape[:2])

	alphamaskzero+= image1[:,:, 3]
	alphamaskzero = alphamaskzero
	alphamaskzero[alphamaskzero==0] = 1
	output = np.zeros((image1.shape[0], image1.shape[1], 3))

	align = np.divide(image1[:, :, 3], alphamaskzero)
	stacked = np.dstack((align, np.dstack((align, align))))
	output += np.multiply(image1[:, :, :3], stacked)
	output = np.clip(output, 0, 1)

	align = np.divide(image2[:, :, 3], alphamaskzero)
	stacked = np.dstack((align, np.dstack((align, align))))
	output += np.multiply(image2[:, :, :3], stacked)
	output = np.clip(output, 0, 1)

	align = np.divide(image3[:, :, 3], alphamaskzero)
	stacked = np.dstack((align, np.dstack((align, align))))
	output += np.multiply(image3[:, :, :3], stacked)
	output = np.clip(output, -1, 1)

	return output
def lap_pyr(image, sig , stack=False):
	layers = []
	layers.append(image)
	for i in range(3):
		layers.append(gaussian(layers[i], sig, mode="constant"))
	layers = np.array(layers)

	if stack:
		return layers
	else:
		for i in range(3):
			layers[i] = layers[i] - layers[i+1]
		return layers

def maskmultiblend(imageA, imageB, imageC, sig):
	
	mask1 = np.zeros(imageA.shape)
	mask2 = np.zeros(imageC.shape)
	mask1[np.any(imageA, 2), :] = 1
	mask2[np.any(imageC, 2), :] = 1
	imask1 = 1 - mask1
	imask2 = 1 - mask2
	iimask1 = gaussian(imask1, sig)
	iimask2 = gaussian(imask2, sig)
	for i in range(3):
		iimask1 += gaussian(imask1, sig)
		iimask2 += gaussian(imask2, sig)
	mask1 = (np.clip(1 - iimask1,0,1))
	mask2 = (np.clip(1 - iimask2,0,1))

	im1 = lap_pyr(imageA, sig)
	base = np.zeros(im1[0].shape)
	im2 = lap_pyr(imageB, sig)
	im3 = lap_pyr(imageC, sig)
	base2 = np.zeros(im3[0].shape)

	lmask = []
	rmask = []
	lmask.append(mask1)
	rmask.append(mask2)
	for i in range(3):
		lmask.append(gaussian(lmask[i], sig, mode="constant"))
		rmask.append(gaussian(rmask[i], sig, mode="constant"))
	lmask = np.array(lmask)
	rmask = np.array(rmask)

	lmask_comp = 1 - lmask
	rmask_comp = 1 - rmask

	im1 = lmask * im1
	im2 = lmask_comp * im2
	im3 = rmask_comp * im3

	for i in range(len(im1)):
		base += (im1 + im2)[i]
	# base = np.clip(base, -1, 1)
	for i in range(len(im3)):
		base2 += (base + im3)[i]
	blend = np.clip(base2, -1, 1)
	return blend




def blender(image1, image2):
	base = np.zeros((image1.shape))
	for a in range(image1.shape[0]):
		for b in range(image1.shape[1]):
			A =  image1[a, b, 0] + image1[a, b, 1] + image1[a, b, 2]
			B =  image2[a, b, 0] + image2[a, b, 1] + image2[a, b, 2]
			if A == B :
				base[a, b, :] = image1[a, b, :] * (1  - .75) + image2[a, b, :] * .75
			else:
				base[a, b, :] = image1[a, b, :] + image2[a, b, :]

	return base



def get_harris_corners(im, edge_discard=20):
    """
    This function takes a b&w image and an optional amount to discard
    on the edge (default is 5 pixels), and finds all harris corners
    in the image. Harris corners near the edge are discarded and the
    coordinates of the remaining corners are returned. A 2d array (h)
    containing the h value of every pixel is also returned.
    h is the same shape as the original image, im.
    coords is 2 x n (ys, xs).
    """

    assert edge_discard >= 20

    # find harris corners
    h = corner_harris(im, method='eps', sigma=1)
    coords = peak_local_max(h, min_distance=15, indices=True)

    # discard points on edge
    edge = edge_discard  # pixels
    mask = (coords[:, 0] > edge) & \
           (coords[:, 0] < im.shape[0] - edge) & \
           (coords[:, 1] > edge) & \
           (coords[:, 1] < im.shape[1] - edge)
    coords = coords[mask].T
    return h, coords

def RANSAC(best_1, best_2):
	
	final_2 = []
	for i in range(1000):
		sam_1 = []
		sam_2 = []
		final = []
		size = range(1, len(best_1[0]))
		rand_4 = rand.sample(size, 4)
		for i in rand_4:
			sam_1.append((best_1[0][i], best_1[1][i]))
			sam_2.append((best_2[0][i], best_2[1][i]))


		sam_1 = np.array(sam_1)
		sam_2 = np.array(sam_2)

		H = computeH(sam_1, sam_2)
	

		err = np.dot(H, np.vstack((best_1, np.ones((1, best_1.shape[1])))))
		base = np.ones(err.shape)
		correct = np.zeros(err.shape)
		# err = err 
		# print("this is error {} ".format(err[2,:]))
		# print(err.shape)
		base[0,:] = err[0,:] / err[2,:] 
		base[1,:] = err[1,:] / err[2,:]
		base[2,:] = err[2,:] / err[2,:]
		correct[0,:] = best_2[0,:]
		correct[1,:] = best_2[1,:]
	# correct[2,:] = best_2[2,:]

	# print(base)
	# print(correct)
	# print(correct[:,0])
		sqe = np.sqrt( np.square(base[0,:] - correct[0,:]) + np.square(base[1,:] - correct[1,:]))
		for i in range(len(best_1[0])):
			if sqe[i] < .6:
				
				final.append(((best_1[0][i], best_1[1][i]) , (best_2[0][i], best_2[1][i])))

		if len(final_2) < len(final):
			final_2 = final
	# print(final_2)

	return final_2
def dist2(x, c):
    """
    dist2  Calculates squared distance between two sets of points.
    Description
    D = DIST2(X, C) takes two matrices of vectors and calculates the
    squared Euclidean distance between them.  Both matrices must be of
    the same column dimension.  If X has M rows and N columns, and C has
    L rows and N columns, then the result has M rows and L columns.  The
    I, Jth entry is the  squared distance from the Ith row of X to the
    Jth row of C.
    Adapted from code by Christopher M Bishop and Ian T Nabney.
    """
    
    ndata, dimx = x.shape
    ncenters, dimc = c.shape
    assert(dimx == dimc, 'Data dimension does not match dimension of centers')

    return (np.ones((ncenters, 1)) * np.sum((x**2).T, axis=0)).T + \
            np.ones((   ndata, 1)) * np.sum((c**2).T, axis=0)    - \
            2 * np.inner(x, c)

def mosaic_idoorpA():
	LP = load_image('./photos/ind1.jpg')
	CP = load_image('./photos/ind2.jpg')
	RP = load_image('./photos/ind3.jpg')

	lp_point = getpoints(LP)
	cp_point = getpoints(CP)
	rp_point = getpoints(RP)

	hlp = computeH(lp_point, cp_point)
	hcp = computeH(cp_point, cp_point)
	hrp = computeH(rp_point, cp_point)


	newimagel = warp(LP, np.linalg.inv(hlp), output_shape=(LP.shape[0]*2, LP.shape[1]*3))
	newimagec = warp(CP, np.linalg.inv(hcp), output_shape=(LP.shape[0]*2, LP.shape[1]*3))
	newimager = warp(RP, np.linalg.inv(hrp), output_shape=(LP.shape[0]*2, LP.shape[1]*3))

#######################################################
	# newimagel = warp(LP, np.linalg.inv(hlp))
	# newimagec = warp(CP, np.linalg.inv(hcp))
	# newimager = warp(RP, np.linalg.inv(hrp))
	mpimg.imsave("./1.jpg", newimagel)
	mpimg.imsave("./2.jpg", newimagec)
	mpimg.imsave("./3.jpg", newimager)
	final = blender(blender(newimagel, newimagec), newimager)
	# result = blender(newimagel, newimagec, newimager)
	mpimg.imsave("./result.jpg", final)
	return

def rectify_chesspA():
	chess_tilted = load_image('./photos/chess7.jpg')
	point_2 = np.array([[367, 800], [625, 800], [625, 1058], [367, 1058]])
	point_3 = np.array([[0, 0], [0, chess_tilted.shape[0]], [chess_tilted.shape[1], chess_tilted.shape[0]]
		, [chess_tilted.shape[0], 0]])
	point_1 = np.array(getpoints(chess_tilted))
	# print(point_1)
	# print(np.array(point_1).T)
	point_1 = np.array(point_1)
	H = computeH(point_1, point_2)
	# H_2 = computeH(point_1, point_3)
	# newimage = warp(chess_tilted, H)
	# mpimg.imsave("./sahba6.jpg", newimage)
	newimage2 = warp(chess_tilted, np.linalg.inv(H))
	mpimg.imsave("./chess.jpg", newimage2)
	return
class PS:
	def __init__(self, score, point):
		self.score = score
		self.point = point
	def scorer(self):
		return self.score
	def pointer(self):
		return self.point
def ANMS_discripor(image, normalize=True):
	gray = rgb2gray(image)
	value, coor = get_harris_corners(gray)
	# c = coor.T
	# print(c.shape)
	# print("x = {} y = {} value = {} ".format(c[0], c[1], value[c[0], c[1]]))

	c = coor.T.tolist()
	list_of_coor = []
	# maxi = 0
	radiues = {}
	print("got the harris points now computing the for loop:\n")
	for main_point in c:
		
		amin = np.array([main_point])
		for other_point in c:
			if(value[main_point[0], main_point[1]] < (.9 * value[other_point[0], other_point[1]])):
			# if the other point has higher value of fxy than our mainpoint we just add it to the lits.
				list_of_coor.append(other_point)
		# list_of_coor.append(np.array(list_of_values))
		while(len(list_of_coor) != 0):

			norm = dist2(amin, np.array(list_of_coor))
			list_of_best = np.amin(norm).astype(float).item()
			radiues[tuple(main_point)] = list_of_best
			
			break

		list_of_coor = []


		# now we wan to find the norm2  between our points and the rest of the points with better score in the
	sort = sorted(radiues.items(), key=lambda x: x[1], reverse=True)
	# print(sort)
	# Now we want to return the best 600 points

		# the list_of_coor and add it to our dict
	# print(sort[:600])
		# once the second loop is done we get the U2 norms compare to the main point
		# # print("x={}, y ={}".format(c[i][1], c[i][0]))
		# list_of_coor.append((c[i][1], c[i][0]))
		# # if c[i][1] > maxi:
		# # 	maxi = c[i][1]x

	best_points = []

###3 you can run this part of the code to get the points on the image.I had to save them  manualy
	for i in range(len(sort[:600])):
		best_points.append((sort[i][0][1], sort[i][0][0]))
	
	best_points = np.array(best_points).T
 
	# print(np.amax(best_points[1]))
	# plt.imshow(image)
	# plt.plot(best_points[0], best_points[1], 'bo')
	# plt.show()

	#  now that we have the 600 points of intrest we now wanna find the discriptors
	# we set out patch size to be 50 therefore for each point we need a offset so we can make that 50 *50 patch
	# the offset should be equal to 25
	def patchmaker(point, image, size):
		patch = np.zeros((size, size))
		for i in range(size):

			for j in range(size):
				
				patch[i][j] = image[point[1] + i][point[0] + j]
				
		return resize(patch, (8 , 8), mode='constant')
	patch_size = 40
	offset = 20
	list_points = []
	patch_dic = {}
	for i in range(len(best_points[0])):
		
		yx_offset = (best_points[0][i] - offset , best_points[1][i] - offset)

		if(yx_offset[1] >= image.shape[0]  - patch_size - 1  or yx_offset[0] >= image.shape[1] - patch_size - 1):

			continue	 
		patch = patchmaker(yx_offset, gray, patch_size)

		if normalize:
			patchnorm = (patch - np.mean(patch))/np.std(patch)
			patch_dic[(best_points[1][i] , best_points[0][i])] = np.reshape(patchnorm, (1, 64))
			list_points.append((best_points[1][i] , best_points[0][i]))
		else:
			patch_dic[(best_points[1][i] , best_points[0][i])] = np.reshape(patchnorm, (1, 64))
			list_points.append((best_points[1][i] , best_points[0][i]))
		# mpimg.imsave("./indoor_batches/ind_batch{}.jpg".format(str(i)), patchnorm)


# now that we have our normalized batches we need to find the matches in the other picter

	return value, c , list_points, patch_dic

def match_finder(points1, key_1, points2, key_2):
	dic = {}
	k = 0
	print("matchfinder loop:\n")
	for i in key_1:
		dic_2 = {}
		for j in key_2:
			dic_2[j] = dist2(points1[i], points2[j])
		dic_2 = sorted(dic_2.items(), key=lambda x: x[1])
		
		# if k is 0:
		# 	print(dic_2)
		# 	print(dic_2[0][0])
		# 	print(dic_2[0][1])
		# 	print(dic_2[1][0])
		# 	print(dic_2[1][1])
		# 	print(dic_2[0][1]/dic_2[1][1])
		if dic_2[0][1]/dic_2[1][1] < .38:
			dic[i] = dic_2[0][0]

		k += 1
	

	return dic
def mosaic_indoor_Auto():
	LP = load_image('./photos/ind1.jpg')
	CP = load_image('./photos/ind2.jpg')
	RP = load_image('./photos/ind3.jpg')
	# value, coor = get_harris_corners(LP_gray)


	# c = coor.T
	# print(c.shape)
	# print("x = {} y = {} value = {} ".format(c[0], c[1], value[c[0], c[1]]))

	# print(coor.T)
	# print(coor.shape[1])
	# numberofcoor = coor.shape[1]
	# plt.imshow(LP)
	# plt.plot(coor[1], coor[0], 'bo')
	# plt.show()

	# we compute the ANMS for each pictuer
	# now that we have our normalized batches we need to find the matches in the other picter
	H_l, coord_l, batchkey,batchnorm = ANMS_discripor(LP)
	H_2, coord_2, batchkey_2, batchnorm_2 = ANMS_discripor(CP)
	H_3, coord_3, batchkey_3, batchnorm_3 = ANMS_discripor(RP)
	matchpoints = match_finder(batchnorm, batchkey, batchnorm_2, batchkey_2)
	matchpoints_2 = match_finder(batchnorm_3, batchkey_3, batchnorm_2, batchkey_2)
	matchpoints_3 = match_finder(batchnorm_2, batchkey_2, batchnorm_2, batchkey_2)
	point1= []
	point2= []
	point2_2= []
	point3= []
	point4 = []
	point5 = []
	for i , j in matchpoints.items():
		
		point1.append(i)
		point2.append(j)
	for i , j in matchpoints_2.items():
		point3.append(i)
		point2_2.append(j)
	for i , j in matchpoints_3.items():
		point4.append(i)
		point5.append(j)

	point1 = np.array(point1).T
	point2 = np.array(point2).T
	point4 = np.array(point4).T
	point5 = np.array(point5).T

	plt.imshow(LP)
	plt.plot(point1[1], point1[0], 'bo')
	plt.show()

	plt.imshow(CP)
	plt.plot(point2[1], point2[0], 'bo')
	plt.show()
	point3 = np.array(point3).T
	point2_2 = np.array(point2_2).T

	final_point = RANSAC(point1, point2)
	Final_point2 = RANSAC(point3, point2_2)
	middle = RANSAC(point2, point2)
	point1= []
	point2= []
	point2_2= []
	point3= []
	middlelist=[]
	point5 = []
	for i in final_point:
		point1.append((i[0][1], i[0][0]))
		point2.append((i[1][1], i[1][0]))
	point1 = np.array(point1).T.astype('float32')
	point2 = np.array(point2).T.astype('float32')
	for i in Final_point2:
		point3.append((i[0][1], i[0][0]))
		point2_2.append((i[1][1], i[1][0]))
	point2_2 = np.array(point2_2).T.astype('float32')
	point3 = np.array(point3).T.astype('float32')
	for i in middle:
		middlelist.append((i[0][1], i[0][0]))
		point5.append((i[1][1], i[1][0]))

	
	H_left = computeH(point1.T, point2.T)
	H_middle = computeH(middlelist, point5)
	H_right = computeH(point3.T.astype('float32'), point2_2.T.astype('float32'))


	plt.imshow(LP)
	plt.plot(point1[0], point1[1], 'bo')
	plt.show()

	plt.imshow(CP)
	plt.plot(point2[0], point2[1], 'bo')
	plt.show()
	
	plt.imshow(RP)
	plt.plot(point3[0], point3[1], 'bo')
	plt.show()

	plt.imshow(CP)
	plt.plot(point2_2[0], point2_2[1], 'bo')
	plt.show()
	newimagel = warp(LP, np.linalg.inv(H_left), output_shape=(LP.shape[0]*2, LP.shape[1]*3))
	newimagec = warp(CP, np.linalg.inv(H_middle), output_shape=(LP.shape[0]*2, LP.shape[1]*3))
	newimager = warp(RP, np.linalg.inv(H_right), output_shape=(LP.shape[0]*2, LP.shape[1]*3))
	mpimg.imsave("./auto_left.jpg", newimagel)
	mpimg.imsave("./auto_middle.jpg", newimagec)
	mpimg.imsave("./auto_right.jpg", newimager)
	newimagel = np.clip(blender(newimagel, newimagec), 0, 1)
	final = blender(newimagel, newimager) 


	mpimg.imsave("./auto_finish.jpg",np.clip(final,0, 1))
	return



def mosaic_city_Auto():
	LP = load_image('./photos/city0.jpg')
	CP = load_image('./photos/city1.jpg')
	RP = load_image('./photos/city2.jpg')
	# value, coor = get_harris_corners(LP_gray)


	# c = coor.T
	# print(c.shape)
	# print("x = {} y = {} value = {} ".format(c[0], c[1], value[c[0], c[1]]))

	# print(coor.T)
	# print(coor.shape[1])
	# numberofcoor = coor.shape[1]
	# plt.imshow(LP)
	# plt.plot(coor[1], coor[0], 'bo')
	# plt.show()

	# we compute the ANMS for each pictuer
	# now that we have our normalized batches we need to find the matches in the other picter
	H_l, coord_l, batchkey,batchnorm = ANMS_discripor(LP)
	H_2, coord_2, batchkey_2, batchnorm_2 = ANMS_discripor(CP)
	H_3, coord_3, batchkey_3, batchnorm_3 = ANMS_discripor(RP)
	matchpoints = match_finder(batchnorm, batchkey, batchnorm_2, batchkey_2)
	matchpoints_2 = match_finder(batchnorm_3, batchkey_3, batchnorm_2, batchkey_2)
	matchpoints_3 = match_finder(batchnorm_2, batchkey_2, batchnorm_2, batchkey_2)
	point1= []
	point2= []
	point2_2= []
	point3= []
	point4 = []
	point5 = []
	for i , j in matchpoints.items():
		
		point1.append(i)
		point2.append(j)
	for i , j in matchpoints_2.items():
		point3.append(i)
		point2_2.append(j)
	for i , j in matchpoints_3.items():
		point4.append(i)
		point5.append(j)

	point1 = np.array(point1).T
	point2 = np.array(point2).T
	point4 = np.array(point4).T
	point5 = np.array(point5).T

	# plt.imshow(LP)
	# plt.plot(point1[1], point1[0], 'bo')
	# plt.show()

	# plt.imshow(CP)
	# plt.plot(point2[1], point2[0], 'bo')
	# plt.show()
	point3 = np.array(point3).T
	point2_2 = np.array(point2_2).T

	final_point = RANSAC(point1, point2)
	Final_point2 = RANSAC(point2_2, point3)
	middle = RANSAC(point2, point2)
	point1= []
	point2= []
	point2_2= []
	point3= []
	middlelist=[]
	point5 = []
	for i in final_point:
		point1.append((i[0][1], i[0][0]))
		point2.append((i[0][1], i[0][0]))
	point1 = np.array(point1).T.astype('float32')
	point2 = np.array(point2).T.astype('float32')
	for i in Final_point2:
		point3.append((i[0][1], i[0][0]))
		point2_2.append((i[0][1], i[0][0]))
	point2_2 = np.array(point2_2).T.astype('float32')
	point3 = np.array(point3).T.astype('float32')
	for i in middle:
		middlelist.append((i[0][1], i[0][0]))
		point5.append((i[0][1], i[0][0]))
	
	H_left = computeH(point1.T, point2.T)
	H_middle = computeH(point2.T, point2.T)
	H_right = computeH(point3.T.astype('float32'), point2_2.T.astype('float32'))

	print(point1)
	plt.imshow(LP)
	plt.plot(point1[0], point1[1], 'bo')
	plt.show()

	plt.imshow(CP)
	plt.plot(point2[0], point2[1], 'bo')
	plt.show()
	
	plt.imshow(RP)
	plt.plot(point3[0], point3[1], 'bo')
	plt.show()

	plt.imshow(CP)
	plt.plot(point2_2[0], point2_2[1], 'bo')
	plt.show()
	newimagel = warp(LP, np.linalg.inv(H_left), output_shape=(LP.shape[0], LP.shape[1]*1.5))
	newimagec = warp(CP, H_middle, output_shape=(CP.shape[0], CP.shape[1]*1.5))
	newimager = warp(RP, H_right, output_shape=(RP.shape[0], RP.shape[1]*1.5))
	mpimg.imsave("./auto_left_city0.jpg", newimagel)
	mpimg.imsave("./auto_middle_city0.jpg", newimagec)
	mpimg.imsave("./auto_right_city0.jpg", newimager)
	newimagel = blender(newimagel, newimagec)
	final = blender(newimagel, newimager)

	mpimg.imsave("./auto_result_city0.jpg", np.clip(final, 0, 1))
	return
