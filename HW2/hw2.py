import cv2
import numpy as np


def sign_lines(img: np.ndarray) -> np.ndarray:
    """
    This function takes in the image as a numpy array and returns a numpy array of lines.

    https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
    :param img: Image as numpy array
    :return: Numpy array of lines.
    """
    # run the sobel edge detection then the hough line transformation

    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #must be in grayscale for hough line detection
    cannyEdge = cv2.Canny(grayscale, threshold1=100, threshold2=2)
    
    theta = np.pi/ 180
    #houghLineP returns cartesian coords while HoughLine returns polar coords
    hough = cv2.HoughLinesP(cannyEdge, 1, theta, threshold=100, minLineLength=50, maxLineGap=10)
    
    return hough

    raise NotImplemented


def sign_circle(img: np.ndarray) -> np.ndarray:
    """
    This function takes in the image as a numpy array and returns a numpy array of circles.
    :param img: Image as numpy array
    :return: Numpy array of circles.
    """

    #upper and lower bounds for traffic light colors
    color_bounds = {
        "red": [(0, 100, 100), (10, 255, 255), (170, 100, 100), (180, 255, 255)],
        "yellow": [(20, 100, 100), (30, 255, 255)],
        "green": [(35, 100, 100), (85, 255, 255)]
    }

    detected_lights = []
    color_map = {"red": 0, "yellow": 1, "green": 2}

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # for color detection
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # must be in grayscale for cannyEdge
    cannyEdge = cv2.Canny(grayscale, threshold1=100, threshold2=200)
    houghCircle = cv2.HoughCircles(cannyEdge, cv2.HOUGH_GRADIENT, dp=1, minDist=25, param1=125, param2=125, minRadius=15, maxRadius=50)
    
    # check the colors of the coords and determine if they are lit or not.
    if houghCircles is not None:
        houghCircles = np.uint16(np.around(houghCircles))
        for circle in houghCircles[0, :]:
            x, y, _ = circle  # Ignore the radius
            hsv_value = hsv[y, x]

            # Determine the color based on HSV ranges
            color = -1
            if any(lower <= hsv_value <= upper for lower, upper in zip(color_bounds["red"][::2], color_bounds["red"][1::2])):
                color = "red"
            elif any(lower <= hsv_value <= upper for lower, upper in zip(color_bounds["yellow"][::2], color_bounds["yellow"][1::2])):
                color = "yellow"
            elif any(lower <= hsv_value <= upper for lower, upper in zip(color_bounds["green"][::2], color_bounds["green"][1::2])):
                color = "green"

            if color:
                detected_lights.append([x, y, color_map[color]])
    
    # Convert detected lights list to np.ndarray
    if detected_lights:
        return np.array(detected_lights)
    else:
        # Return an empty array if no circles were detected
        return np.array([], dtype=int).reshape(0, 3)

    raise NotImplemented

    #examples of output
    # np.array([[x1, y1, 0], [x2, y2, 1], [x3, y3, 2]])  (Red, yellow, and green detected)
    # np.array([[x1, y1, -1], [x2, y2, -1], [x3, y3, -1]])  (All detected as unlit)




def sign_axis(lines: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    This function takes in a numpy array of lines and returns a tuple of np.ndarray and np.ndarray.

    This function should identify the lines that make up a sign and split the x and y coordinates.
    :param lines: Numpy array of lines.
    :return: Tuple of np.ndarray and np.ndarray with each np.ndarray consisting of the x coordinates and y coordinates
             respectively.
    """
    xaxis = np.empty(0, dtype=np.int32)
    yaxis = np.empty(0, dtype=np.int32)
    xaxis = []
    yaxis = []

    for line in lines:
        xaxis.extend([line[0], line[2]]) # adding x1, and x2
        yaxis.extend([line[1], line[3]]) # adding y1 and y2
    return xaxis, yaxis


def identify_traffic_light(img: np.ndarray) -> tuple:
    """
    This function takes in the image as a numpy array and returns a tuple identifying the location
    of the traffic light in the image and the lighted light.
    :param img: Image as numpy array
    :return: Tuple identifying the location of the traffic light in the image and light.
             ( x,   y, color)
             (140, 100, 'None') or (140, 100, 'Red')
             In the case of no light lit, coordinates can be just center of traffic light
    """
    lines = sign_lines(img)
    circles = sign_circle(img)
    x_coords, y_coords = sign_axis(lines)

    center_x = int(np.mean(x_coords)) if x_coords.size > 0 else 0
    center_y = int(np.mean(y_coords)) if y_coords.size > 0 else 0

    # Determine the light color or if unlit based on sign_circle output
    light_status = []
    if circles.size > 0:
        for x, y, color_code in circles:
            if color_code == 0:
                color = "Red"
            elif color_code == 1:
                color = "Yellow"
            elif color_code == 2:
                color = "Green"
            else:
                color = "None"  # -1 code or no color detected
            
            # Append each detected light's status and coordinates
            light_status.append((x, y, color))
    
    # If no lights are lit or detected, return central coordinates with 'None'
    if not light_status or all(status[2] == "None" for status in light_status):
        return (center_x, center_y, "None")
    
    # Otherwise, return the coordinates and color status of each detected light
    return light_status

    raise NotImplemented


def identify_stop_sign(img: np.ndarray) -> tuple:
    """
    This function takes in the image as a numpy array and returns a tuple of the sign location and name.
    :param img: Image as numpy array
    :return: tuple with x, y, and sign name
             (x, y, 'stop')
    """
    raise NotImplemented


def identify_yield(img: np.ndarray) -> tuple:
    """
    This function takes in the image as a numpy array and returns a tuple of the sign location and name.
    :param img: Image as numpy array
    :return: tuple with x, y, and sign name
             (x, y, 'yield')
    """
    raise NotImplemented


def identify_construction(img: np.ndarray) -> tuple:
    """
    This function takes in the image as a numpy array and returns a tuple of the sign location and name.
    :param img: Image as numpy array
    :return: tuple with x, y, and sign name
             (x, y, 'construction')
    """
    raise NotImplemented


def identify_warning(img: np.ndarray) -> tuple:
    """
    This function takes in the image as a numpy array and returns a tuple of the sign location and name.
    :param img: Image as numpy array
    :return: tuple with x, y, and sign name
             (x, y, 'warning')
    """
    raise NotImplemented


def identify_rr_crossing(img: np.ndarray) -> tuple:
    """
    This function takes in the image as a numpy array and returns a tuple of the sign location and name.
    :param img: Image as numpy array
    :return: tuple with x, y, and sign name
             (x, y, 'rr_crossing')
    """
    raise NotImplemented


def identify_services(img: np.ndarray) -> tuple:
    """
    This function takes in the image as a numpy array and returns a tuple of the sign location and name.
    :param img: Image as numpy array
    :return: tuple with x, y, and sign name
             (x, y, 'services')
    """
    raise NotImplemented


def identify_signs(img: np.ndarray) -> np.ndarray:
    """
    This function takes in the image as a numpy array and returns a numpy array of all signs locations and name.
    Call the other identify functions to determine where that sign is if it exists.
    :param img: Image as numpy array
    :return: Numpy array of all signs locations and name.
             [[x, y, 'stop'],
              [x, y, 'construction']]
    """
    raise NotImplemented


def identify_signs_noisy(img: np.ndarray) -> np.ndarray:
    """
    This function takes in the image as a numpy array and returns a numpy array of all signs locations and name.
    Call the other identify functions to determine where that sign is if it exists.

    The images will have gaussian noise applied to them so you will need to do some blurring before detection.
    :param img: Image as numpy array
    :return: Numpy array of all signs locations and name.
             [[x, y, 'stop'],
              [x, y, 'construction']]
    """
    raise NotImplemented


def identify_signs_real(img: np.ndarray) -> np.ndarray:
    """
    This function takes in the image as a numpy array and returns a numpy array of all signs locations and name.
    Call the other identify functions to determine where that sign is if it exists.

    The images will be real images so you will need to do some preprocessing before detection.
    You may also need to adjust existing functions to detect better with real images through named parameters
    and other code paths

    :param img: Image as numpy array
    :return: Numpy array of all signs locations and name.
             [[x, y, 'stop'],
              [x, y, 'construction']]
    """
    raise NotImplemented