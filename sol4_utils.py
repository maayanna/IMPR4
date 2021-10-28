from scipy.signal import convolve2d
import numpy as np
import imageio
import skimage.color
from math import factorial as fac
import scipy.signal as sig
import scipy.ndimage as ndi


NORMALIZE = 255
DIM_MIN = 16

def binomial(x, y):
    binom = fac(x) // fac(y) // fac(x - y)
    return binom

def pascal(m):
    return [binomial(m - 1, y) for y in range(m)]

def create_filter_vec(binomes):
    filter_vec = list()
    k = len(binomes)
    numerator = 1/2**(k-1)
    for bin in binomes:
        filter_vec.append(bin*numerator)
    return np.array(filter_vec)

def read_image(filename, representation):
    """
    Function that reads an image file and convert it into a given representation
    :param filename: the filename of an image on disk
    :param representation: representation code, either 1 or 2 defining whether the output should
                           be a grayscale image (1) or an RGB image (2)
    :return: an image represented by a matrix of type np.float64
    """

    color_flag = True #if RGB image
    image = imageio.imread(filename)

    float_image = image.astype(np.float64)

    if not np.all(image <= 1):
        float_image /= NORMALIZE #Normalized to range [0,1]

    if len(float_image.shape) != 3 : #Checks if RGB or Grayscale
        color_flag = False

    if color_flag and representation == 1 : #Checks if need RGB to Gray
        return skimage.color.rgb2gray(float_image)

    # Same coloring already
    return float_image


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    This function build a Gaussian pyramid
    :param im: a grayscale image with double values in [0,1]
    :param max_levels: the maximum number of levels in the resulting pyramid
    :param filter_size: the size of the gaussian filter (an odd scalar that represents a squared filter) to
                        be used in constructing the pyramid filter
    :return: pyr, filter_vec
    """
    binom_vec = create_filter_vec(pascal(filter_size))
    filter_vec = binom_vec.reshape(1, filter_size)
    filter = sig.convolve2d(filter_vec, filter_vec.T)

    pyramid = list()
    pyramid.append(im)

    for index in range(max_levels - 1):
        blured = ndi.filters.convolve(im, filter)
        blured = blured[::2,::2]


        height, width = blured.shape
        if height < DIM_MIN or width < DIM_MIN:
            break

        pyramid.append(blured)
        # plt.imshow(blured, cmap='gray')
        # plt.show()
        im = blured

    return pyramid, filter_vec

def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img
