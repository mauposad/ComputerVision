import cv2
import numpy as np
import os

# Load ArUco dictionary and detector parameters for marker detection
# Most of this is from my old images for better detection, I just left it as is
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50) # 6x6 ArUco markers with 50 bits
parameters = cv2.aruco.DetectorParameters() # Detector parameters for ArUco markers
parameters.adaptiveThreshWinSizeMin = 3 # Adaptive thresholding window size
parameters.adaptiveThreshWinSizeMax = 150 # Adaptive thresholding window size
parameters.adaptiveThreshWinSizeStep = 10 # Adaptive thresholding window size
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX # Corner refinement method
parameters.perspectiveRemoveIgnoredMarginPerCell = 0.3 # Perspective removal ignored margin per cell
parameters.maxErroneousBitsInBorderRate = 0.5 # Maximum erroneous bits in border rate
parameters.minMarkerPerimeterRate = 0.02 # Minimum marker perimeter rate
parameters.maxMarkerPerimeterRate = 4.0 # Maximum marker perimeter rate

detector = cv2.aruco.ArucoDetector(aruco_dict, parameters) # ArUco marker detector

# Load ArUco marker reference images
# I have 4 markers in the images folder to overlay the overlay.jpeg on the base image/video
aruco_marker_paths = {
    0: 'aruco1.jpeg',
    1: 'aruco2.jpeg',
    2: 'aruco3.jpeg',
    3: 'aruco4.jpeg'
}

# This function will overlay the overlay image on the base image using the detected points
def overlay_image(base_img, overlay_img, src_pts): # src_pts are the detected points in the base image
    h, w = overlay_img.shape[:2] # Height and width of the overlay image
    dst_pts = np.array([[w, 0], [0, 0], [0, h], [w, h]], dtype=np.float32) # Destination points for overlay image
    matrix, _ = cv2.findHomography(dst_pts, src_pts) # Find homography matrix from destination to source points
    warped_overlay = cv2.warpPerspective(overlay_img, matrix, (base_img.shape[1], base_img.shape[0])) # Warp overlay image
    mask = np.zeros_like(base_img, dtype=np.uint8) # Create mask for overlay image
    cv2.fillConvexPoly(mask, src_pts.astype(int), (255, 255, 255)) # Fill mask with detected points
    base_masked = cv2.bitwise_and(base_img, cv2.bitwise_not(mask)) # Mask the base image
    combined = cv2.add(base_masked, warped_overlay) # Add the overlay image to the base image
    return combined

# Function to order points in required order: top-right -> top-left -> bottom-left -> bottom-right
def order_points_custom(pts): # pts are the detected points
    pts = sorted(pts, key=lambda x: x[0]) # Sort points based on x-coordinates
    
    # Split into left-most and right-most
    left_pts = sorted(pts[:2], key=lambda x: x[1])  # Top-left and bottom-left sorted by y
    right_pts = sorted(pts[2:], key=lambda x: x[1])  # Top-right and bottom-right sorted by y
    return np.array([right_pts[0], left_pts[0], left_pts[1], right_pts[1]], dtype=np.float32) # Return ordered points

# Input and output folders
input_folder = 'input_media'
output_folder = 'output_media'
os.makedirs(output_folder, exist_ok=True)

# Images and overlay
images = ['image0.jpeg', 'image1.jpeg', 'image2.jpeg', 'image3.jpeg'] 
overlay_img_path = os.path.join(input_folder, 'photowithJeremy.jpg') # "No, I am your father"

# Check if overlay image exists - "I find your lack of faith disturbing"
if not os.path.isfile(overlay_img_path):
    print(f"Error: Overlay image not found at {overlay_img_path}")
    exit()

overlay_img = cv2.imread(overlay_img_path) # Read overlay image

# Process images
for img_file in images: # Loop through images
    img_path = os.path.join(input_folder, img_file) # Image path
    if not os.path.isfile(img_path): # Check if image exists
        print(f"Error: Image not found at {img_path}")
        continue

    # Read image and convert to grayscale for marker detection
    base_img = cv2.imread(img_path) # Read image
    gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY) # Convert to grayscale
    corners, ids, _ = detector.detectMarkers(gray) # Detect markers in the image

    # Check if markers are detected and at least 4 markers are detected
    if ids is not None and len(ids) >= 4:
        id_list = ids.flatten() # Flatten IDs because they are in a list of lists (lol)
        detected_pts = [] # Detected points list
        for i in range(4): # Loop through 4 markers
            if i in id_list: # Check if marker ID is in the list
                idx = np.where(ids == i)[0][0] # Get index of marker ID
                detected_pts.append(corners[idx][0].mean(axis=0)) # Append detected points to list

        # Check if 4 markers are detected
        if len(detected_pts) == 4: 
            ordered_pts = order_points_custom(detected_pts) # Order detected points
            try: # Try to overlay the overlay image on the base image
                result = overlay_image(base_img, overlay_img, ordered_pts) # Overlay image
                output_file = os.path.join(output_folder, f'output_{img_file}') # Output file path
                cv2.imwrite(output_file, result) # Save output image
                print(f"Saved: {output_file}") # Print saved message (I have a bad feeling about this)
            except ValueError as e:  # Catch any errors 
                print(f"Error processing {img_file}: {e}") 
        else:
            print(f"Not enough markers detected in {img_file}. Detected points: {len(detected_pts)}") # debug message (MORE))
    else:
        print(f"Markers not detected in {img_file}") 

# Process video
video_input = os.path.join(input_folder, 'ardunoVideo.mp4') 
video_overlay_input = os.path.join(input_folder, 'recurse_video.mp4')
video_output = os.path.join(output_folder, 'output_video.mp4')

# Check if both videos exist
if not os.path.isfile(video_input):
    print(f"Error: Video not found at {video_input}") 
    exit()

if not os.path.isfile(video_overlay_input):
    print(f"Error: Overlay video not found at {video_overlay_input}") 
    exit()

# Open the original and overlay videos
cap = cv2.VideoCapture(video_input)
overlay_cap = cv2.VideoCapture(video_overlay_input)

# Get video properties dynamically
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Video Properties:\n- Width: {frame_width}\n- Height: {frame_height}\n- FPS: {fps}") 

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_output, fourcc, fps, (frame_width, frame_height))

# Process video frames
while cap.isOpened() and overlay_cap.isOpened():
    ret, frame = cap.read()
    overlay_ret, overlay_frame = overlay_cap.read()

    if not ret or not overlay_ret:
        print("End of video or no more frames to read.") 
        break
    
    # Convert frame to grayscale for marker detection 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray) 

    # Check if markers are detected and at least 4 markers are detected
    if ids is not None and len(ids) >= 4:
        id_list = ids.flatten()
        detected_pts = []
        for i in range(4):
            if i in id_list:
                idx = np.where(ids == i)[0][0]
                detected_pts.append(corners[idx][0].mean(axis=0))

        # Overlay if all 4 points are detected
        if len(detected_pts) == 4:
            ordered_pts = order_points_custom(detected_pts)
            try:
                frame = overlay_image(frame, overlay_frame, ordered_pts)
            except ValueError as e:
                print(f"Error processing video frame: {e}")
        else:
            print(f"Not enough markers detected in video frame. Detected points: {len(detected_pts)}")
    else:
        print("Markers not detected in video frame.")
    
    out.write(frame)

cap.release()
overlay_cap.release()
out.release()
print(f"Video saved: {video_output}")