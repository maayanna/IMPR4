# Initial code for ex4.
# You may change this code, but keep the functions' signatures
# You can also split the code to multiple files as long as this file's API is unchanged

import numpy as np
import os
import matplotlib.pyplot as plt
import skimage.data
import skimage.color

from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass
import shutil
from imageio import imwrite

import scipy.signal as sig
import scipy.ndimage as ndi

import sol4_utils


MATRIX = np.array([[1,0,-1]])
KERNEL_SIZE = 3
K = 0.04

def harris_corner_detector(im):
    """
    Detects harris corners.
    Make sure the returned coordinates are x major!!!
    :param im: A 2D array representing an image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """

    #Deriatives Ix and Iy
    dx = sig.convolve2d(im, MATRIX, mode="same", boundary="symm")
    dy = sig.convolve2d(im, MATRIX.T, mode="same", boundary="symm")

    #Blurring images
    dx2_blurred = sol4_utils.blur_spatial(dx**2, KERNEL_SIZE)
    dy2_blurred = sol4_utils.blur_spatial(dy**2, KERNEL_SIZE)
    dx_dy = dx*dy
    dx_dy_blurred = sol4_utils.blur_spatial(dx_dy, KERNEL_SIZE)

    #matrix M = ( (dx**2, dy*dy) , (dy*dx, dy**2) ) -> det(M) = (( dx * dy) - (dx*dy)**2 )
    det = ((dx2_blurred)*(dy2_blurred)) - (dx_dy_blurred**2)
    trace = dx2_blurred + dy2_blurred

    r = det - (K*(trace**2))

    #Binary image that contains the local maximum points
    im_max_points = non_maximum_suppression(r)

    return np.array(np.nonzero(im_max_points.T)).T


def sample_descriptor(im, pos, desc_rad):
    """
    Samples descriptors at the given corners.
    :param im: A 2D array representing an image.
    :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
    """

    k = 1 + (2*desc_rad)
    my_descriptor = np.empty(shape=(pos.shape[0], k, k))
    # print(pos)
    for index, position in enumerate(pos):

        first = np.arange(position[0] - desc_rad, position[0] + desc_rad + 1)
        secd = np.arange(position[1] - desc_rad, position[1] + desc_rad + 1)
        new_grid = np.meshgrid(first, secd)

        map_coordinates = ndi.map_coordinates(im, new_grid[::-1], order=1, prefilter=False)
        map_coordinates -= np.mean(map_coordinates)

        window_norm = np.linalg.norm(map_coordinates)

        if window_norm: #norm different from 0
            map_coordinates /= window_norm

        my_descriptor[index, :, :] = map_coordinates

    return my_descriptor


def find_features(pyr):
    """
    Detects and extracts feature points from a pyramid.
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
    :return: A list containing:
                1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                   These coordinates are provided at the pyramid level pyr[0].
                2) A feature descriptor array with shape (N,K,K)
    """

    points = spread_out_corners(pyr[0], 7, 7, 3)
    #sending pyramid level 3
    my_descriptor = sample_descriptor(pyr[2], points/4, 3) #because taking level 3 (smaller picture)
    return points, my_descriptor


def match_features(desc1, desc2, min_score):
    """
    Return indices of matching descriptors.
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :param min_score: Minimal match score.
    :return: A list containing:
                1) An array with shape (M,) and dtype int of matching indices in desc1.
                2) An array with shape (M,) and dtype int of matching indices in desc2.
    """

    desc1 = desc1.reshape(desc1.shape[0], desc1.shape[1]**2)
    desc2 = desc2.reshape(desc2.shape[0], desc2.shape[1]**2)
    match_score = np.dot(desc1, desc2.T)

    #Wat to take the second bigger, at place -2

    big_row = np.partition(match_score, -2, 1)[:, -2]
    big_col = np.partition(match_score, -2, 0)[-2, :]


    #Now checking the conditions

    conds = np.transpose((match_score.T>=big_row))&(match_score>=big_col)&(match_score>min_score)
    return np.where(conds)


def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
    """

    new_array = np.column_stack((pos1, np.ones(len(pos1), dtype=np.float64)))
    to_normalize = np.dot(H12, new_array.T).T
    to_normalize[:,:2] /= (to_normalize[:, 2].reshape(pos1.shape[0],1))
    return np.delete(to_normalize, -1, axis=1)


def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
    """
    Computes homography between two sets of points using RANSAC.
    :param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
    :param pos2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: A list containing:
                1) A 3x3 normalized homography matrix.
                2) An Array with shape (S,) where S is the number of inliers,
                    containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """

    my_inliers = list()

    for index in range(num_iter):

        #Finding the transformation
        rand_index = np.random.choice(np.arange(points1.shape[0]), size=2)
        matrix_transform = estimate_rigid_transform(points1[rand_index], points2[rand_index], translation_only)

        # if matrix_transform is not None: # If transformation, needs to apply homography
        after_homog = apply_homography(points1, matrix_transform)
        euclidean_dist = np.linalg.norm(after_homog - points2, axis=1)**2

        try_inliers = np.where(euclidean_dist < inlier_tol)[0]


        if len(try_inliers) > len(my_inliers):
            my_inliers = try_inliers

    first = points1[my_inliers]
    scd = points2[my_inliers]

    return estimate_rigid_transform(first, scd, translation_only), my_inliers

def display_matches(im1, im2, points1, points2, inliers):
    """
    Dispalay matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :parma pos1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
    :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """

    pictures = np.hstack((im1, im2))
    plt.imshow(pictures, cmap='gray')
    plt.plot(points1[:,0], points1[:,1],'o', color="red",markersize = '1')
    points2[:,0] += im1.shape[1] #shift
    plt.plot(points2[:,0], points2[:,1],'o', color="red",markersize = '1')
    plt.plot((points1[:,0], points2[:,0]), (points1[:,1], points2[:,1]),color = "blue", lw = '0.2')
    plt.plot((points1[inliers][:,0], points2[inliers][:,0]), (points1[inliers][:,1], points2[inliers][:,1]),color = "yellow", lw = '0.5')
    plt.show()


def accumulate_homographies(H_succesive, m):
    """
    Convert a list of succesive homographies to a
    list of homographies to a common reference frame.
    :param H_successive: A list of M-1 3x3 homography
      matrices where H_successive[i] is a homography which transforms points
      from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to
      accumulate the given homographies.
    :return: A list of M 3x3 homography matrices,
      where H2m[i] transforms points from coordinate system i to coordinate system m
    """
    array_return = list(H_succesive)
    array_return[m] = np.eye(3)  #replace the homography matrix with the id matrix for image index m


    for index in range(m-1, -1, -1): #from m-1 to 0
        array_return[index] = np.dot(array_return[index+1], H_succesive[index])
        array_return[index] /= array_return[index][2, 2]  # H[2,2] == 1

    for index in range(m+1, len(H_succesive)): #from m+1 to len(H_successive) till the end
        to_compute = np.linalg.inv(H_succesive[index])
        array_return[index] = np.dot(array_return[index-1], to_compute)
        array_return[index] /= array_return[index][2,2] # H[2,2] == 1

    #the last one to add to stitch it all

    array_return.append(np.dot(array_return[-1], np.linalg.inv(H_succesive[-1]))) #need to nornalize ?

    return array_return


def compute_bounding_box(homography, w, h):
    """
    computes bounding box of warped image under homography, without actually warping the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the top left corner,
     and the second row is the [x,y] of the bottom right corner
    """

    corners = np.array([[0,0], [w, 0], [w, h], [0, h]])
    homography_applied = apply_homography(corners, homography)

    top_left = [np.min(homography_applied[:,0]), np.min(homography_applied[:,1])]
    bottom_right = [np.max(homography_applied[:,0]), np.max(homography_applied[:,1])]

    return np.array([top_left, bottom_right]).astype(np.int)


def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homograhpy.
    :return: A 2d warped image.
    """

    top_left, bottom_right = compute_bounding_box(homography, image.shape[1], image.shape[0])

    x = np.arange(top_left[0], bottom_right[0])
    y = np.arange(top_left[1], bottom_right[1])

    first, second = np.meshgrid(x, y)

    coordinates = np.zeros((2, first.reshape(-1).shape[0]))
    coordinates[0, :] = first.reshape(-1)
    coordinates[1, :] = second.reshape(-1)

    homography_applied1, homography_applied2 = apply_homography(coordinates.T, np.linalg.inv(homography)).T
    final_image = ndi.map_coordinates(image, [homography_applied2, homography_applied1], order=1, prefilter=False)
    return final_image.reshape((len(y), len(x)))


def warp_image(image, homography):
    """
    Warps an RGB image with a given homography.
    :param image: an RGB image.
    :param homography: homograhpy.
    :return: A warped image.
    """
    return np.dstack([warp_channel(image[..., channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
    """
    Filters rigid transformations encoded as homographies by the amount of translation from left to right.
    :param homographies: homograhpies to filter.
    :param minimum_right_translation: amount of translation below which the transformation is discarded.
    :return: filtered homographies..
    """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if homographies[i][0, -1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
    """
    Computes rigid transforming points1 towards points2, using least squares method.
    points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
    :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
    :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
    :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
    :return: A 3x3 array with the computed homography.
    """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2, :2] = rotation
    H[:2, 2] = translation
    return H


def non_maximum_suppression(image):
    """
    Finds local maximas of an image.
    :param image: A 2D array representing an image.
    :return: A boolean array with the same shape as the input image, where True indicates local maximum.
    """
    # Find local maximas.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max() * 0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
    centers = np.stack(centers).round().astype(np.int)
    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret


def spread_out_corners(im, m, n, radius):
    """
    Splits the image im to m by n rectangles and uses harris_corner_detector on each.
    :param im: A 2D array representing an image.
    :param m: Vertical number of rectangles.
    :param n: Horizontal number of rectangles.
    :param radius: Minimal distance of corner points from the boundary of the image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    corners = [np.empty((0, 2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:, 0] > radius) & (corners[:, 0] < im.shape[1] - radius) &
             (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
    ret = corners[legit, :]
    return ret


class PanoramicVideoGenerator:
    """
    Generates panorama from a set of images.
    """

    def __init__(self, data_dir, file_prefix, num_images):
        """
        The naming convention for a sequence of images is file_prefixN.jpg,
        where N is a running number 001, 002, 003...
        :param data_dir: path to input images.
        :param file_prefix: see above.
        :param num_images: number of images to produce the panoramas with.
        """
        self.file_prefix = file_prefix
        self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
        self.files = list(filter(os.path.exists, self.files))
        self.panoramas = None
        self.homographies = None
        print('found %d images' % len(self.files))

    def align_images(self, translation_only=False):
        """
        compute homographies between all images to a common coordinate system
        :param translation_only: see estimte_rigid_transform
        """
        # Extract feature point locations and descriptors.
        points_and_descriptors = []
        for file in self.files:
            image = sol4_utils.read_image(file, 1)
            self.h, self.w = image.shape
            pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
            points_and_descriptors.append(find_features(pyramid))

        # Compute homographies between successive pairs of images.
        Hs = []
        for i in range(len(points_and_descriptors) - 1):
            points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
            desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

            # Find matching feature points.
            ind1, ind2 = match_features(desc1, desc2, .7)
            points1, points2 = points1[ind1, :], points2[ind2, :]

            # Compute homography using RANSAC.
            H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

            # Uncomment for debugging: display inliers and outliers among matching points.
            # In the submitted code this function should be commented out!
            # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

            Hs.append(H12)

        # Compute composite homographies from the central coordinate system.
        accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
        self.homographies = np.stack(accumulated_homographies)
        self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
        self.homographies = self.homographies[self.frames_for_panoramas]

    def generate_panoramic_images(self, number_of_panoramas):
        """
        combine slices from input images to panoramas.
        :param number_of_panoramas: how many different slices to take from each input image
        """
        assert self.homographies is not None

        # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
        self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

        # change our reference coordinate system to the panoramas
        # all panoramas share the same coordinate system
        global_offset = np.min(self.bounding_boxes, axis=(0, 1))
        self.bounding_boxes -= global_offset

        slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
        warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
        # every slice is a different panorama, it indicates the slices of the input images from which the panorama
        # will be concatenated
        for i in range(slice_centers.size):
            slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
            # homography warps the slice center to the coordinate system of the middle image
            warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
            # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
            warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

        panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

        # boundary between input images in the panorama
        x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
        x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                      x_strip_boundary,
                                      np.ones((number_of_panoramas, 1)) * panorama_size[0]])
        x_strip_boundary = x_strip_boundary.round().astype(np.int)

        self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
        for i, frame_index in enumerate(self.frames_for_panoramas):
            # warp every input image once, and populate all panoramas
            image = sol4_utils.read_image(self.files[frame_index], 2)
            warped_image = warp_image(image, self.homographies[i])
            x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
            y_bottom = y_offset + warped_image.shape[0]

            for panorama_index in range(number_of_panoramas):
                # take strip of warped image and paste to current panorama
                boundaries = x_strip_boundary[panorama_index, i:i + 2]
                image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
                x_end = boundaries[0] + image_strip.shape[1]
                self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

        # # crop out areas not recorded from enough angles
        # # assert will fail if there is overlap in field of view between the left most image and the right most image
        crop_left = int(self.bounding_boxes[0][1, 0])
        crop_right = int(self.bounding_boxes[-1][0, 0])
        assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
        print(crop_left, crop_right)
        self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

    def save_panoramas_to_video(self):
        assert self.panoramas is not None
        out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
        try:
            shutil.rmtree(out_folder)
        except:
            print('could not remove folder')
            pass
        os.makedirs(out_folder)
        # save individual panorama images to 'tmp_folder_for_panoramic_frames'
        for i, panorama in enumerate(self.panoramas):
            imwrite('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
        if os.path.exists('%s.mp4' % self.file_prefix):
            os.remove('%s.mp4' % self.file_prefix)
        # write output video to current folder
        os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
                  (out_folder, self.file_prefix))

    def show_panorama(self, panorama_index, figsize=(20, 20)):
        assert self.panoramas is not None
        plt.figure(figsize=figsize)
        plt.imshow(self.panoramas[panorama_index].clip(0, 1))
        plt.show()



#if __name__=="__main__":
#     im = skimage.color.rgb2gray(skimage.data.astronaut())
    # plt.imshow(im, cmap='gray')
    # plt.show()
    # x = harris_corner_detector(im)
    # print(x)
    # plt.imshow(im, cmap='gray')
    # plt.plot(x[:,0],x[:,1],'o')
    # plt.show()
    # im = np.arange(start=0, stop=100, dtype=np.float64).reshape(10,10)
    # pos = np.array([[5,5],[3,4]])
    # print(im)
    # print(sample_descriptor(im, pos, 3))
    # im1 = sol4_utils.read_image("external/oxford001.jpg", 1)
    # im2 = sol4_utils.read_image("external/oxford002.jpg", 1)
    # pyr1,filter = sol4_utils.build_gaussian_pyramid(im1,3,3)
    # pyr2,filter_ = sol4_utils.build_gaussian_pyramid(im2,3,3)
    # feat1,desc1 = find_features(pyr1)
    # feat2,desc2 = find_features(pyr2)
    # print(feat1.shape)
    # print(desc1.shape)
    # print(feat2.shape)
    # print(desc2.shape)
    #
    # print()
    #
    # print(desc1)
    # print()
    # points1, points2 = match_features(desc1,desc2,0.5)
    # h12, inliers = ransac_homography(feat1[points1], feat2[points2],100, 100, False)
    # display_matches(im1,im2,feat1[points1],feat2[points2],inliers)
    #
    # x = PanoramicVideoGenerator("external", "oxford", 2)
    # x.align_images()
    # x.generate_panoramic_images(1)
    # x.show_panorama(0)