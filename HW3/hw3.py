import cv2
import numpy as np
import os

# Function to find the center of a marker
def findCenter(corner: np.array) -> np.array:
    corner_new = corner[0]
    x_center = np.mean(corner_new[:, 0])
    y_center = np.mean(corner_new[:, 1])
    return np.array([x_center, y_center])

# Function to process a single frame or image
def processFrame(frame, overlay_img):
    """
    Detects ArUco markers in a frame or image and overlays an image.

    Args:
        frame: Current video frame or input image.
        overlay_img: Image to overlay between markers.

    Returns:
        Processed frame with the overlay applied, or the original frame if not enough markers are detected.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load ArUco dictionary and detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # Detect markers
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None and len(ids) >= 4:
        print(f"Detected marker IDs: {ids.flatten()}")

        # Store marker centers by ID
        marker_centers = {}
        for i in range(len(ids)):
            center = findCenter(corners[i])
            marker_centers[ids[i][0]] = center

        # Verify that all required IDs are detected (e.g., 0, 1, 2, 3)
        required_ids = [0, 1, 2, 3]
        if all(req_id in marker_centers for req_id in required_ids):
            top_left = marker_centers[0]
            top_right = marker_centers[1]
            bottom_left = marker_centers[2]
            bottom_right = marker_centers[3]

            # Define points for perspective transform
            pts_base = np.float32([top_left, top_right, bottom_left, bottom_right])
            pts_warp = np.float32([[0, 0], [400, 0], [0, 400], [400, 400]])

            # Compute the homography
            perspective = cv2.getPerspectiveTransform(pts_warp, pts_base)

            # Warp the overlay image
            warped_img = cv2.warpPerspective(
                overlay_img,
                perspective,
                (frame.shape[1], frame.shape[0])
            )

            # Create a mask for the warped image
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [pts_base.astype(np.int32)], 255)

            # Blend the warped image with the base frame
            mask_img = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
            finished_frame = cv2.add(mask_img, warped_img)

            return finished_frame
        else:
            print(f"Not all required IDs detected. Detected IDs: {ids.flatten()}")
            return frame
    else:
        print("Not enough markers detected in this frame.")
        return frame

# Function to test with static images
def testWithImages(image_paths, overlay_img_path, results_folder):
    """
    Tests the overlay logic with a list of static images and saves the results.

    Args:
        image_paths: List of paths to input images.
        overlay_img_path: Path to the overlay image.
        results_folder: Path to save the processed images.
    """
    overlay_img = cv2.imread(overlay_img_path)
    if overlay_img is None:
        print("Error: Overlay image not found.")
        return

    overlay_img = cv2.resize(overlay_img, (400, 400))

    for idx, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            continue

        processed_image = processFrame(image, overlay_img)

        # Save the processed image
        result_path = os.path.join(results_folder, f"processed_image_{idx+1}.jpg")
        cv2.imwrite(result_path, processed_image)
        print(f"Saved processed image: {result_path}")

        # Optionally display the processed image
        cv2.imshow(f"Processed Image - {image_path}", processed_image)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to process video
def processVideo(video_path, overlay_path, output_path, results_folder):
    """
    Processes a video frame-by-frame to overlay an image based on ArUco markers.

    Args:
        video_path: Path to the input video.
        overlay_path: Path to the overlay image.
        output_path: Path to save the processed video.
        results_folder: Path to save the processed frames.
    """
    # Load the overlay image
    overlay_img = cv2.imread(overlay_path)
    if overlay_img is None:
        print("Error: Overlay image not found.")
        return

    # Resize overlay image
    overlay_img = cv2.resize(overlay_img, (400, 400))

    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Initialize the video writer
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Process the current frame
        processed_frame = processFrame(frame, overlay_img)

        # Save the processed frame
        result_frame_path = os.path.join(results_folder, f"frame_{frame_idx+1:04d}.jpg")
        cv2.imwrite(result_frame_path, processed_frame)
        print(f"Saved processed frame: {result_frame_path}")

        # Write the processed frame to the output video
        out.write(processed_frame)

        # Display the processed frame (optional)
        cv2.imshow("Processed Video", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    video.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processed video saved to: {output_path}")

# Main script
if __name__ == "__main__":
    # Create results folder
    results_folder = "results"
    os.makedirs(results_folder, exist_ok=True)

    # Test with static images
    image_paths = ["aruco11.jpeg", "aruco12.jpeg"]  
    overlay_path = "photowithJeremy.jpg"  
    testWithImages(image_paths, overlay_path, results_folder)

    # Process the video
    video_path = "video.mp4"  
    output_path = os.path.join(results_folder, "output_video.mp4")
    processVideo(video_path, overlay_path, output_path, results_folder)
