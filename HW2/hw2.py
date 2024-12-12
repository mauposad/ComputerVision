import cv2
import numpy as np


def sign_lines(img: np.ndarray) -> np.ndarray:
    """
    This function takes in the image as a numpy array and returns a numpy array of lines. 

    https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
    :param img: Image as numpy array
    :return: Numpy array of lines.
    """
    
    copy_img = np.copy(img) # Create a copy of the image
    
    gray = cv2.cvtColor(copy_img, cv2.COLOR_BGR2GRAY) # grayscale it
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) # Apply Gaussian blur
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150) # Apply Canny edge detection

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(
        edges, 
        rho=1,
        theta=np.pi / 180, 
        threshold=50, 
        minLineLength=30, 
        maxLineGap=10)

    if lines is None: # If no lines detected, return an empty array
        return np.array([])
    return lines


def sign_circle(img: np.ndarray) -> np.ndarray:
    """
    This function takes in the image as a numpy array and returns a numpy array of circles.
    :param img: Image as numpy array
    :return: Numpy array of circles.
    """
    copy_img = np.copy(img) # Create a copy of the image
    gray = cv2.cvtColor(copy_img, cv2.COLOR_BGR2GRAY) # grayscale it
    blurred = cv2.GaussianBlur(gray, (9, 9), 2) # Apply Gaussian blur

    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(
        blurred,
        method=cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=100
    )
    
    circles = np.round(circles[0, :]).astype("int") # Convert the (x, y, r) values to integers

    if circles is None: # If no circles are detected, return an empty array
        return np.array([])
    return circles    


def sign_axis(lines: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    This function takes in a numpy array of lines and returns a tuple of np.ndarray and np.ndarray.

    This function should identify the lines that make up a sign and split the x and y coordinates.
    :param lines: Numpy array of lines.
    :return: Tuple of np.ndarray and np.ndarray with each np.ndarray consisting of the x coordinates and y coordinates
             respectively.
    """
    # Initialize empty arrays for x and y coordinates
    xaxis = np.empty(0, dtype=np.int32)
    yaxis = np.empty(0, dtype=np.int32)

    # Check if lines is None or empty
    if lines is None or len(lines) == 0:
        return xaxis, yaxis

    # Extract x and y coordinates from the lines
    for line in lines:
        x1, y1, x2, y2 = line[0] # Unpack the line coordinates
        xaxis = np.append(xaxis, [x1, x2]) # Append x coordinates
        yaxis = np.append(yaxis, [y1, y2]) # Append y coordinates

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
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Convert to HSV color space

    # Define HSV color ranges for Red, Yellow, and Green
    red_lower1 = np.array([0, 120, 120])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 120, 120])
    red_upper2 = np.array([180, 255, 255])
    yellow_lower = np.array([15, 115, 115]) # fixed yellow range
    yellow_upper = np.array([35, 255, 255]) # fixed yellow range
    green_lower = np.array([40, 120, 120])
    green_upper = np.array([70, 255, 255])

    # Create masks for each color
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1) # two masks for red to account for wrap around in HSV
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2) # I guess that makes sense
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)  # Combine red masks 
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper) 
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    # Initialize variables to store results
    traffic_light_center = (0, 0) 
    light_color = 'None'

    # Detect circles (lights) using Hough Circle Transform
    masks = [('Red', red_mask), ('Yellow', yellow_mask), ('Green', green_mask)]
    for color, mask in masks:
        blurred = cv2.GaussianBlur(mask, (5, 5), 0)
        circles = cv2.HoughCircles( 
            blurred,
            method=cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=50
        )
        
        # If circles are detected, get the center and radius of the circle
        if circles is not None: 
            x, y, r = map(int, circles[0][0])
            traffic_light_center = (x, y)
            light_color = color
            break

    return (*traffic_light_center, light_color) # Return the center and color of the light


def identify_stop_sign(img: np.ndarray) -> tuple:
    """
    This function takes in the image as a numpy array and returns a tuple of the sign location and name.
    :param img: Image as numpy array
    :return: tuple with x, y, and sign name
             (x, y, 'stop')
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Convert to HSV color space

    # Define HSV range for red (two ranges to account for wrap around in HSV)
    red_lower1 = np.array([0, 100, 100]) 
    red_upper1 = np.array([10, 255, 255]) 
    red_lower2 = np.array([160, 100, 100]) 
    red_upper2 = np.array([179, 255, 255]) 

    # Create masks for red 
    mask1 = cv2.inRange(hsv, red_lower1, red_upper1) 
    mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(mask1, mask2) # Combine red masks

    contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Find contours in the mask

    # Iterate through contours and find the stop sign
    for contour in contours:
        perimeter = cv2.arcLength(contour, True) # Calculate the perimeter of the contour
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True) # Approximate the contour to a polygon

        # Check if the polygon has 8 sides (octagon)
        if len(approx) == 8:
            # Calculate the center of the contour
            M = cv2.moments(contour)
            if M["m00"] != 0: # Check if the contour has area
                cX = int(M["m10"] / M["m00"]) 
                cY = int(M["m01"] / M["m00"])
                return cX, cY, "stop" # Return the center and tag

    return 0, 0, "None" # If no stop sign is found, return default values


def identify_yield(img: np.ndarray) -> tuple:
    """
    This function takes in the image as a numpy array and returns a tuple of the sign location and name.
    :param img: Image as numpy array
    :return: tuple with x, y, and sign name
             (x, y, 'yield')
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Convert to HSV color space

    # Define HSV range for red (two ranges to account for wrap around in HSV) 
    red_lower1 = np.array([0, 100, 100])
    red_upper1 = np.array([10, 255, 255]) 
    red_lower2 = np.array([160, 100, 100])
    red_upper2 = np.array([179, 255, 255]) 

    # Create masks for red
    mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Find contours in the mask

    # Iterate through contours to find the yield sign 
    for contour in contours:
        perimeter = cv2.arcLength(contour, True) # Calculate the perimeter of the contour
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True) # Approximate the contour to a polygon

        # Check if the polygon has 3 sides (triangle)
        if len(approx) == 3:
            # Calculate the center of the contour
            M = cv2.moments(contour) 
            if M["m00"] != 0: # Check if the contour has area
                cX = int(M["m10"] / M["m00"]) 
                cY = int(M["m01"] / M["m00"])
                return cX, cY, "yield"

    return 0, 0, "None" # If no yield sign is found, return default values


def identify_construction(img: np.ndarray) -> tuple:
    """
    This function takes in the image as a numpy array and returns a tuple of the sign location and name.
    :param img: Image as numpy array
    :return: tuple with x, y, and sign name
             (x, y, 'construction')
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Convert to HSV color space

    # Define HSV range for orange
    orange_lower = np.array([10, 100, 100]) 
    orange_upper = np.array([25, 255, 255]) # fixed orange range?

    orange_mask = cv2.inRange(hsv, orange_lower, orange_upper) # Create a mask for orange
    
    contours, _ = cv2.findContours(orange_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Find contours in the mask

    # Iterate through contours to find the construction sign
    for contour in contours:
        perimeter = cv2.arcLength(contour, True) # Calculate the perimeter of the contour
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True) # Approximate the contour to a polygon

        # Check if the polygon has 4 sides (diamond shape) because construction signs are diamond shaped apparently
        if len(approx) == 4:
            # Ensure the shape is approximately square (aspect ratio near 1)
            x, y, w, h = cv2.boundingRect(approx) # Get the bounding rectangle of the contour
            aspect_ratio = float(w) / h # Calculate the aspect ratio
            if 0.9 <= aspect_ratio <= 1.1: # okay so it's square-ish
                # Calculate the center of the contour
                M = cv2.moments(contour) 
                if M["m00"] != 0: # Check if the contour has area
                    cX = int(M["m10"] / M["m00"]) 
                    cY = int(M["m01"] / M["m00"])
                    return cX, cY, "construction" # Return the center and tag
                
    return 0, 0, "None" # If no construction sign is found, return default values


def identify_warning(img: np.ndarray) -> tuple:
    """
    This function takes in the image as a numpy array and returns a tuple of the sign location and name.
    :param img: Image as numpy array
    :return: tuple with x, y, and sign name
             (x, y, 'warning')
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Convert to HSV color space

    # Define HSV range for yellow
    yellow_lower = np.array([20, 100, 100]) # hmmm maybe fix yellow range
    yellow_upper = np.array([30, 255, 255])

    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper) # Create a mask for yellow

    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Find contours in the mask 

    # Iterate through contours to find the warning sign
    for contour in contours:
        perimeter = cv2.arcLength(contour, True) # Calculate the perimeter of the contour
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True) # Approximate the contour to a polygon

        # Check if the polygon has 4 sides (diamond shape) because warning signs are also diamond shaped
        if len(approx) == 4:
            # Ensure the shape is approximately square (aspect ratio near 1)
            x, y, w, h = cv2.boundingRect(approx) # Get the bounding rectangle of the contour
            aspect_ratio = float(w) / h # Calculate the aspect ratio
            if 0.9 <= aspect_ratio <= 1.1: # yeah yeah square 
                # Calculate the center of the contour
                M = cv2.moments(contour) 
                if M["m00"] != 0: # Check if the contour has area
                    cX = int(M["m10"] / M["m00"]) 
                    cY = int(M["m01"] / M["m00"])
                    return cX, cY, "warning"
                
    return 0, 0, "None" # If no warning sign is found, return default values


def identify_rr_crossing(img: np.ndarray) -> tuple:
    """
    This function takes in the image as a numpy array and returns a tuple of the sign location and name.
    :param img: Image as numpy array
    :return: tuple with x, y, and sign name
             (x, y, 'rr_crossing')
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Convert to HSV color space

    # Define HSV range for yellow
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])

    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper) # Create a mask for yellow
    blurred_mask = cv2.GaussianBlur(yellow_mask, (5, 5), 0) # Apply Gaussian blur to the mask

    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(
        blurred_mask,
        cv2.HOUGH_GRADIENT, 
        dp=1.2,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=20,
        maxRadius=100
    )

    if circles is not None:
        # Use the first detected circle
        x, y, r = map(int, circles[0][0]) 
        return x, y, "rr_crossing" # Return the center and tag

    return 0, 0, "None" # If no railroad crossing sign is found, return default values


def identify_services(img: np.ndarray) -> tuple:
    """
    This function takes in the image as a numpy array and returns a tuple of the sign location and name.
    :param img: Image as numpy array
    :return: tuple with x, y, and sign name
             (x, y, 'services')
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Convert to HSV color space

    # Define HSV range for blue
    blue_lower = np.array([100, 150, 100])  # Lower blue range 
    blue_upper = np.array([130, 255, 255])  # Upper blue range

    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper) # Create a mask for blue
    
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Find contours in the mask

    # Iterate through contours to find the services sign
    for contour in contours:
        perimeter = cv2.arcLength(contour, True) # Calculate the perimeter of the contour
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True) # Approximate the contour to a polygon

        # Check if the polygon has 4 sides (rectangular shape)
        if len(approx) == 4:
            # Ensure the shape is a rectangle (aspect ratio can vary)
            x, y, w, h = cv2.boundingRect(approx) # Get the bounding rectangle of the contour
            aspect_ratio = float(w) / h # Calculate the aspect ratio
            if 0.5 <= aspect_ratio <= 2.0:  # Broad range for rectangular shapes
                # Calculate the center of the contour
                M = cv2.moments(contour) 
                if M["m00"] != 0: # Check if the contour has area
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    return cX, cY, "services" # Return the center and tag

    return 0, 0, "None" # If no services sign is found, return default values


def identify_signs(img: np.ndarray) -> np.ndarray:
    """
    This function takes in the image as a numpy array and returns a numpy array of all signs locations and name.
    Call the other identify functions to determine where that sign is if it exists.
    :param img: Image as numpy array
    :return: Numpy array of all signs locations and name.
             [[x, y, 'stop'],
              [x, y, 'construction']]
    """
    detected_signs = [] # Initialize a list to store detected signs

    # Call individual sign identification functions and append the results to the list of detected signs
    stop_sign = identify_stop_sign(img)
    if stop_sign[2] != "None": 
        detected_signs.append(stop_sign)

    construction_sign = identify_construction(img)
    if construction_sign[2] != "None":
        detected_signs.append(construction_sign)

    yield_sign = identify_yield(img)
    if yield_sign[2] != "None":
        detected_signs.append(yield_sign)

    rr_crossing_sign = identify_rr_crossing(img)
    if rr_crossing_sign[2] != "None":
        detected_signs.append(rr_crossing_sign)

    services_sign = identify_services(img)
    if services_sign[2] != "None":
        detected_signs.append(services_sign)

    warning_sign = identify_warning(img)
    if warning_sign[2] != "None":
        detected_signs.append(warning_sign)

    return np.array(detected_signs, dtype=object) # Convert the list of detected signs to a numpy array 


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
    blurred_img = cv2.GaussianBlur(img, (5, 5), 1.5) # Apply Gaussian blur to reduce noise

    detected_signs = [] # Initialize a list to store detected signs

    # Call individual sign identification functions with the blurred image and append the results to the list of detected signs
    stop_sign = identify_stop_sign(blurred_img) 
    if stop_sign[2] != "None": 
        detected_signs.append(stop_sign)

    construction_sign = identify_construction(blurred_img)
    if construction_sign[2] != "None":
        detected_signs.append(construction_sign)

    yield_sign = identify_yield(blurred_img)
    if yield_sign[2] != "None":
        detected_signs.append(yield_sign)

    rr_crossing_sign = identify_rr_crossing(blurred_img)
    if rr_crossing_sign[2] != "None":
        detected_signs.append(rr_crossing_sign)

    services_sign = identify_services(blurred_img)
    if services_sign[2] != "None":
        detected_signs.append(services_sign)

    warning_sign = identify_warning(blurred_img)
    if warning_sign[2] != "None":
        detected_signs.append(warning_sign)

    return np.array(detected_signs, dtype=object) # Convert the list of detected signs to a numpy array


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
    blurred_img = cv2.GaussianBlur(img, (3, 3), 0) # Apply Gaussian blur to reduce noise

    hsv_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV) # Convert to HSV color space

    # Perform histogram equalization on the V (value) channel to enhance contrast and improve detection
    hsv_img[:, :, 2] = cv2.equalizeHist(hsv_img[:, :, 2])
    preprocessed_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

    detected_signs = [] # Initialize a list to store detected signs

    stop_sign = identify_stop_sign(preprocessed_img) 
    if stop_sign[2] != "None":
        detected_signs.append(stop_sign)

    construction_sign = identify_construction(preprocessed_img)
    if construction_sign[2] != "None":
        detected_signs.append(construction_sign)

    yield_sign = identify_yield(preprocessed_img)
    if yield_sign[2] != "None":
        detected_signs.append(yield_sign)

    rr_crossing_sign = identify_rr_crossing(preprocessed_img)
    if rr_crossing_sign[2] != "None":
        detected_signs.append(rr_crossing_sign)

    services_sign = identify_services(preprocessed_img)
    if services_sign[2] != "None":
        detected_signs.append(services_sign)

    warning_sign = identify_warning(preprocessed_img)
    if warning_sign[2] != "None":
        detected_signs.append(warning_sign)

    return np.array(detected_signs, dtype=object) # Convert the list of detected signs to a numpy array
