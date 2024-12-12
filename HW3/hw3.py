import cv2
import numpy as np

# Function to find the center of a marker
def findCenter(corner: np.array) -> np.array:
    """
    Calculate the center of a marker based on its four corners.

    Args:
        corner (np.array): Array containing the corners of the marker.

    Returns:
        np.array: The (x, y) coordinates of the center of the marker.
    """
    corner_new = corner[0]
    x_center = np.mean(corner_new[:, 0])
    y_center = np.mean(corner_new[:, 1])
    return np.array([x_center, y_center])

# Function to override IDs based on spatial arrangement
def assignIDsByPosition(corners):
    """
    Assign IDs to markers based on their spatial arrangement (top-left, top-right, etc.).

    Args:
        corners (list): List of detected marker corners.

    Returns:
        dict: Dictionary mapping overridden IDs (0, 1, 2, 3) to marker centers.
    """
    # Find centers of all detected markers
    marker_centers = [findCenter(corner) for corner in corners]

    # Sort centers into top-left, top-right, bottom-left, bottom-right
    # Sort by y-coordinate first (row-wise), then x-coordinate (column-wise)
    sorted_centers = sorted(marker_centers, key=lambda x: (x[1], x[0]))

    # Assign positions to IDs
    return {
        0: sorted_centers[0],  # Top-left
        1: sorted_centers[1],  # Top-right
        2: sorted_centers[2],  # Bottom-left
        3: sorted_centers[3],  # Bottom-right
    }


# Function to process a single image
def processImage(image, overlay_img):
    """
    Detects ArUco markers, overrides IDs by position, and overlays an image.

    Args:
        image: Input image with ArUco markers.
        overlay_img: Image to overlay between markers.

    Returns:
        Resultant image with the overlay applied, or None if not enough markers.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load ArUco dictionary and detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # Detect markers
    corners, ids, _ = detector.detectMarkers(gray)

    # Visualize detected markers for debugging
    if corners is not None:
        print("Markers detected. Overriding IDs based on position.")
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        cv2.imshow("Detected Markers", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if corners is not None and len(corners) == 4:
        # Override IDs based on spatial arrangement
        overridden_ids = assignIDsByPosition(corners)

        try:
            top_left = overridden_ids[0]
            top_right = overridden_ids[1]
            bottom_left = overridden_ids[2]
            bottom_right = overridden_ids[3]

            # Define points for perspective transform
            pts_base = np.float32([top_left, top_right, bottom_left, bottom_right])
            pts_warp = np.float32([[0, 0], [400, 0], [0, 400], [400, 400]])

            # Compute the homography
            perspective = cv2.getPerspectiveTransform(pts_base, pts_warp)

            # Warp the overlay image
            warped_img = cv2.warpPerspective(
                overlay_img,
                perspective,
                (image.shape[1], image.shape[0])
            )

            # Create a mask for the warped image
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [pts_base.astype(np.int32)], 255)

            # Blend the warped image with the base image
            mask_img = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
            finished_img = cv2.add(mask_img, warped_img)

            return finished_img
        except IndexError:
            print("Error: Not enough valid markers for homography!")
            return image
    else:
        print("Not enough markers detected.")
        return None
    

# Main function to process all images and overlay onto video frames
def main():
    # Load the images
    wall_images = [cv2.imread(f"image{i}.jpeg") for i in range(11, 13)]
    # marker_images = [cv2.imread(f"aruco{i}.jpeg") for i in range(1, 5)]
    overlay_img = cv2.imread("photowithJeremy.jpg")

    # Resize overlay image to match expected warp dimensions
    overlay_img = cv2.resize(overlay_img, (400, 400))

    # Process each wall image
    processed_images = [processImage(img, overlay_img) for img in wall_images]

    # Load video
    video = cv2.VideoCapture("viceo.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (int(video.get(3)), int(video.get(4))))

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # Process each frame with the overlay from the first processed image
        frame_with_overlay = processImage(frame, processed_images[0])

        # Write the frame to the output video
        out.write(frame_with_overlay)

        # Display the frame (optional)
        cv2.imshow("Overlay Video", frame_with_overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    video.release()
    out.release()
    cv2.destroyAllWindows()

# Run the main function
if __name__ == "__main__":
    main()
