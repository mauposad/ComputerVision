import cv2
import numpy as np


def read_image(image_path: str) -> np.ndarray:
    """
    This function reads an image and returns it as a numpy array
    :param image_path: String of path to file
    :return img: Image array as ndarray
    """

    img = cv2.imread(image_path)

    # check if image is succesfully loaded
    if img is None:
        raise FileNotFoundError("Image at path {image_path} could not be found")
    

    return img

     # raise NotImplementedError


def extract_green(img: np.ndarray) -> np.ndarray:
    """
    This function takes an image and returns the green channel
    :param img: Image array as ndarray
    :return: Image array as ndarray of just green channel
    """
    img_copy = img.copy()
    green_channel = img[:,:,1]
    return green_channel

    raise NotImplementedError


def extract_red(img: np.ndarray) -> np.ndarray:
    """
    This function takes an image and returns the red channel
    :param img: Image array as ndarray
    :return: Image array as ndarray of just red channel
    """
    img_copy = img.copy()
    red_channel = img[:,:,2]
    return red_channel

    raise NotImplementedError


def extract_blue(img: np.ndarray) -> np.ndarray:
    """
    This function takes an image and returns the blue channel
    :param img: Image array as ndarray
    :return: Image array as ndarray of just blue channel
    """
    img_copy = img.copy()
    blue_channel = img_copy[:,:,0]
    return blue_channel

    raise NotImplementedError


def swap_red_green_channel(img: np.ndarray) -> np.ndarray:
    """
    This function takes an image and returns the image with the red and green channel
    :param img: Image array as ndarray
    :return: Image array as ndarray of red and green channels swapped
    """

    #swap img to to avoid messing up original
    swapped_img = img.copy()


    # swap the red (index 2) and green (index 1) channel
    swapped_img[:,:,[1,2]] = swapped_img[:,:,[2,1]]
    return swapped_img

    raise NotImplementedError


def embed_middle(img1: np.ndarray, img2: np.ndarray, embed_size: (int, int)) -> np.ndarray:
    """
    This function takes two images and embeds the embed_size pixels from img2 onto img1
    :param img1: Image array as ndarray
    :param img2: Image array as ndarray
    :param embed_size: Tuple of size (width, height)
    :return: Image array as ndarray of img1 with img2 embedded in the middle
    """
    img1_copy = img1.copy()
    img2_copy = img2.copy()


    #img 1 & img2 dimensions
    height1, width1 = img1_copy.shape[:2]
    height2, width2 = img2_copy.shape[:2]

    #calculate center for embedding for img 1
    start_x1 = (width1- embed_size[0]) // 2
    start_y1 = (height1 - embed_size[1]) // 2

    # Calculate the center position for extraction from img2
    start_x2 = (width2 - embed_size[0]) // 2
    start_y2 = (height2 - embed_size[1]) // 2

    # Extract the 60x60 region from the middle of img2
    extracted_region = img2[start_y2:start_y2 + embed_size[1], start_x2:start_x2 + embed_size[0]]

    # Embed the extracted region in the middle of img1
    img1_copy[start_y1:start_y1 + embed_size[1], start_x1:start_x1 + embed_size[0]] = extracted_region


    return img1_copy

    raise NotImplementedError


def calc_stats(img: np.ndarray) -> np.ndarray:
    """
    This function takes an image and returns the mean and standard deviation
    :param img: Image array as ndarray
    :return: Numpy array with mean and standard deviation in that order
    """

    mean, stddev = cv2.meanStdDev(img)

    #flatten in order to return it as one array. (2 rows)
    result = np.vstack((mean.flatten(), stddev.flatten()))

    return result

    raise NotImplementedError


def shift_image(img: np.ndarray, shift_val: int) -> np.ndarray:
    """
    This function takes an image and returns the image shifted by shift_val pixels to the right.
    Should have an appropriate border for the shifted area:
    https://docs.opencv.org/3.4/dc/da3/tutorial_copyMakeBorder.html

    Returned image should be the same size as the input image.
    :param img: Image array as ndarray
    :param shift_val: Value to shift the image
    :return: Shifted image as ndarray
    """

    img_copy = img.copy()
    height, width = img.shape[:2] 

    M = np.float32([[1,0, shift_val], [0, 1, 0]])
    warped_image = cv2.warpAffine(img_copy, M, (height, width))
    # cv2.imshow("warped image", warped_image)
    # cv2.waitKey(0)
    # border_image = cv2.copyMakeBorder(warped_image, 0, 0, 0, shift_val, cv2.BORDER_REPLICATE)

    return warped_image 
    raise NotImplementedError


def difference_image(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    This function takes two images and returns the first subtracted from the second

    Make sure the image to return is normalized:
    https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga87eef7ee3970f86906d69a92cbf064bd

    :param img1: Image array as ndarray
    :param img2: Image array as ndarray
    :return: Image array as ndarray
    """

    # Need to resize image due to differences in dimensions
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    new_img = img1 - img2_resized

    normalized_img = cv2.normalize(new_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype= -1)
    return normalized_img

    raise NotImplementedError


def add_channel_noise(img: np.ndarray, channel: int, sigma: int) -> np.ndarray:
    """
    This function takes an image and adds noise to the specified channel.

    Should probably look at randn from numpy

    Make sure the image to return is normalized:
    https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga87eef7ee3970f86906d69a92cbf064bd

    :param img: Image array as ndarray
    :param channel: Channel to add noise to
    :param sigma: Gaussian noise standard deviation
    :return: Image array with gaussian noise added
    """

    mean = 0
    gaussian_noise = np.random.normal(mean, sigma, img.shape).astype(np.float32)

    noisy_img = img.copy()
    noisy_img[:, :, channel] = np.clip(noisy_img[:,:,channel] + gaussian_noise, 0, 255)
    raise NotImplementedError


def add_salt_pepper(img: np.ndarray) -> np.ndarray:
    """
    This function takes an image and adds salt and pepper noise.

    Must only work with grayscale images
    :param img: Image array as ndarray
    :return: Image array with salt and pepper noise
    """
    raise NotImplementedError


def blur_image(img: np.ndarray, ksize: int) -> np.ndarray:
    """
    This function takes an image and returns the blurred image

    https://docs.opencv.org/4.x/dc/dd3/tutorial_gausian_median_blur_bilateral_filter.html
    :param img: Image array as ndarray
    :param ksize: Kernel Size for medianBlur
    :return: Image array with blurred image
    """
    raise NotImplementedError