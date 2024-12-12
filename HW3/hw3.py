import cv2
import numpy as np
from cv2.aruco import detectMarkers
from numpy.array_api import float32



def findCenter(corner: np.array)->np.array:

    #lengthy process but bulls the corners apart into grid locations
    corner_new = corner[0]
    c_1 = corner_new[0]
    c_2 = corner_new[1]
    c_3 = corner_new[2]
    c_4 = corner_new[3]
    x1,y1 = c_1[0],c_1[1]
    x2,y2 = c_2[0],c_2[1]
    x3,y3 = c_3[0],c_3[1]
    x4,y4 = c_4[0],c_4[1]

    #breaks the center from 8 to 4 parts
    
    x_new_center1 = (x1+x2)/2
    x_new_center2 = (x3+x4)/2
    y_new_center1 = (y1+y2)/2
    y_new_center2 = (y3+y4)/2

    #breaks the center from 4 to a two part grid
    x_center = (x_new_center1+x_new_center2)/2
    y_center = (y_new_center1+y_new_center2)/2

    return x_center,y_center



def main ()->None:


    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

    # ## MAKING MARKER 1
    # marker1_id = 250
    # marker1_size = 200
    # marker1 = cv2.aruco.generateImageMarker(aruco_dict, marker1_size, marker1_id)

    # cv2.imwrite('marker_41.jpg', marker1)
    # ## MAKING MARKER 2
    # marker2_id = 250
    # marker2_size = 205
    # marker2 = cv2.aruco.generateImageMarker(aruco_dict, marker2_size, marker2_id)

    # cv2.imwrite('marker_42.jpg', marker2)
    # ## MAKING MARKER 3
    # marker3_id = 250
    # marker3_size = 210
    # marker3 = cv2.aruco.generateImageMarker(aruco_dict, marker3_size, marker3_id)

    # cv2.imwrite('marker_43.jpg', marker3)
    # ## MAKING MARKER 4
    # marker4_id = 250
    # marker4_size = 215
    # marker4 = cv2.aruco.generateImageMarker(aruco_dict, marker4_size, marker4_id)

    # cv2.imwrite('markser_44.jpg', marker4)


    img1 = cv2.imread('aruco1.jpeg')
    # img2 = cv2.imread('image2.jpeg')
    # img3 = cv2.imread('image3.jpeg')
    # img4 = cv2.imread('image3.jpeg')
    all_4_markers = cv2.imread('image3.jpeg')
    video = cv2.VideoCapture('ardunoVideo.mp4')
    img_overlay = cv2.imread('photowithJeremy.jpg')
    #
    # cv2.imshow('img1', img1)
    # cv2.imshow('img2', img2)
    # cv2.waitKey(0)




    # # # Convert the image to grayscale
    # gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    # parameters = cv2.aruco.DetectorParameters()
    # # Create the ArUco detector
    # detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    # # Detect the markers
    # corners, ids, rejected = detector.detectMarkers(gray)

    # #Print the detected markers
    # print("Detected markers:", ids)
    # if ids is not None:
    #     cv2.aruco.drawDetectedMarkers(img1, corners, ids)
    #     cv2.imshow('Detected Markers', img1)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()




#    ##Finding 2
#     gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#     aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
#     parameters = cv2.aruco.DetectorParameters()
#     # Create the ArUco detector
#     detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
#     # Detect the markers
#     corners, ids, rejected = detector.detectMarkers(gray2)

#     # Print the detected markers
#     print("Detected markers:", ids)
#     if ids is not None:
#         cv2.aruco.drawDetectedMarkers(img2, corners, ids)
#         cv2.imshow('Detected Markers', img2)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()




#     ## Finding 3
#     gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
#     aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
#     parameters = cv2.aruco.DetectorParameters()
#     # Create the ArUco detector
#     detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
#      # Detect the markers
#     corners, ids, rejected = detector.detectMarkers(gray3)

#     # Print the detected markers
#     print("Detected markers:", ids)
#     if ids is not None:
#         cv2.aruco.drawDetectedMarkers(img3, corners, ids)
#         cv2.imshow('Detected Markers', img3)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()




    # ## Finding 4
    # gray4 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    # parameters = cv2.aruco.DetectorParameters()
    # # Create the ArUco detector
    # detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    # # Detect the markers
    # corners, ids, rejected = detector.detectMarkers(gray)

    # # Print the detected markers
    # print("Detected markers:", ids)
    # if ids is not None:
    #     cv2.aruco.drawDetectedMarkers(img4, corners, ids)
    #     cv2.imshow('Detected Markers', img4)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()





    ## Attemping to detect all 4 on a screen
    gray = cv2.cvtColor(all_4_markers, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    # Create the ArUco detector
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    # Detect the markers
    corners, ids, rejected = detector.detectMarkers(gray)
    # Print the detected markers
    print("Detected markers:\n", ids)
    corners_1 = corners[0]
    corners_2 = corners[1]
    corners_3 = corners[2]
    corners_4 = corners[3]
    cent_1 = findCenter(corners_1)
    cent_2 = findCenter(corners_2)
    cent_3 = findCenter(corners_3)
    cent_4 = findCenter(corners_4)

    ### Warping Images
    pts_base = np.float32([[cent_1[0],cent_1[1]],[cent_2[0],cent_2[1]],[cent_3[0],cent_3[1]],[cent_4[0],cent_4[1]]])
    pts_warp = np.float32([[0,0],[0,400],[400,0],[400,400]])
    perspective = cv2.getPerspectiveTransform(pts_warp, pts_base)
    warped_img = cv2.warpPerspective(img_overlay, perspective, (all_4_markers.shape[1], all_4_markers.shape[0]))##backwards?
    ###mask needed
    mask = np.zeros(all_4_markers.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [pts_base.astype(np.int32)], (0, 0, 0))

    # Blend the warped image onto the base image
    mask_img= cv2.bitwise_and(all_4_markers,cv2.bitwise_not(mask))
    finished_img = cv2.add(mask_img, warped_img)

    # cv2.imshow("Result", finished_img)
    # cv2.waitKey(0)

    if ids is not None:
            cv2.aruco.drawDetectedMarkers(finished_img, corners, ids)
            cv2.imshow("Result", finished_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()



    #
    # ##### TRYING TO OVERLAY IMAGE ON IMAGE FIRST
    #
    # index_point = np.squeeze(np.where(ids==200))
    # ref_pt1 = np.squeeze(corners[index_point[0]])[1]
    #
    # index_point = np.squeeze(np.where(ids==205))
    # ref_pt2 = np.squeeze(corners[index_point[0]])[2]
    #
    #  ## Finding and securing reference points based off aruco detection
    #
    #
    #
    ## Video Capturing
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))

    size = (frame_width, frame_height)
    output= cv2.VideoWriter('overlay_output.avi',cv2.VideoWriter_fourcc(*'MJPG'),10, size)

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        parameters = cv2.aruco.DetectorParameters()
            # Create the ArUco detector
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            # Detect the markers
        corners, ids, rejected = detector.detectMarkers(gray)
        ##Getting moving corners
        corners_1 = corners[0]
        corners_2 = corners[1]
        corners_3 = corners[2]
        corners_4 = corners[3]
        cent_1 = findCenter(corners_1)
        cent_2 = findCenter(corners_2)
        cent_3 = findCenter(corners_3)
        cent_4 = findCenter(corners_4)
            # Print the detected markers
        # print("Detected markers:", ids)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            ### Warping Images
            pts_base_frame = np.float32(
                [[cent_1[0], cent_1[1]], [cent_2[0], cent_2[1]], [cent_3[0], cent_3[1]], [cent_4[0], cent_4[1]]])
            pts_warp_frame = np.float32([[0,0],[0,400],[400,0],[400,400]])
            perspective = cv2.getPerspectiveTransform(pts_warp_frame, pts_base_frame)
            warped_frame = cv2.warpPerspective(img_overlay, perspective, (frame.shape[1], frame.shape[0]))  ##backwards?
            ###mask needed
            mask = np.zeros(frame.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [pts_base.astype(np.int32)], (0, 0, 0))

            # Blend the warped image onto the base image
            mask_frame = cv2.bitwise_and(frame, cv2.bitwise_not(mask))
            finished_frame = cv2.add(mask_frame, warped_frame)
            finished_frame = cv2.bitwise_or(finished_frame, mask)



        #print(frame[173, 391]) ## Displays the images
        output.write(finished_frame)
        cv2.imshow('mask', finished_frame)
        cv2.waitKey(1)


        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    video.release()
    output.release()
    cv2.destroyAllWindows()

    # pass

if __name__ == '__main__':
    main()