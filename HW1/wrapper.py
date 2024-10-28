import os
import cv2
from hw1 import *


def main() -> None:
    # TODO: Add in images to read
    img1 = read_image("HW1/images/young-juniper.jpg")
    img2 = read_image("HW1/images/old-juniper.jpg")


    # TODO: replace None with the correct code to convert img1 and img2
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    img1_red = extract_red(img1)
    img1_green = extract_green(img1)
    img1_blue = extract_blue(img1)
    
    img2_red = extract_red(img2)
    img2_green = extract_green(img2)
    img2_blue = extract_blue(img2)


    img1_swap = swap_red_green_channel(img1)
    img2_swap = swap_red_green_channel(img2)


    embed_img = embed_middle(img1, img2, (60, 60))

    img1_stats = calc_stats(img1_gray)
    img2_stats = calc_stats(img2_gray)

    # TODO: Replace None with correct calls
    img1_shift = shift_image(img1, 50)
    img2_shift = shift_image(img2, 50)



    # TODO: Replace None with correct calls. The difference should be between the original and shifted image
    img1_diff = difference_image(img1, img2)
    img2_diff = difference_image(img2, img1)
    # TODO: Select appropriate sigma and call functions
    sigma = 50
    img1_noise_red = add_channel_noise(img1, 2, sigma)
    img1_noise_green = add_channel_noise(img1, 1, sigma)
    img1_noise_blue = add_channel_noise(img1, 0, sigma)

    img2_noise_red = add_channel_noise(img2, 2, sigma)
    img2_noise_green = add_channel_noise(img2, 1, sigma)
    img2_noise_blue = add_channel_noise(img2, 0, sigma)

    img1_spnoise = add_salt_pepper(img1_gray)
    img2_spnoise = add_salt_pepper(img2_gray)

    # TODO: Select appropriate ksize, must be odd
    ksize = 5
    img_blur = blur_image(img1_spnoise, ksize)
    img2_blur = blur_image(img2_spnoise, ksize)



    # TODO: Write out all images to appropriate files


if __name__ == '__main__':
    main()