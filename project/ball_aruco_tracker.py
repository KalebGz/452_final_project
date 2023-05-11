#!/usr/bin/env python

# import the necessary packages
from imutils.video import VideoStream
import pyrealsense2 as rs
import numpy as np
import argparse
import time
import cv2
import sys
import csv

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", type=str,
    default="DICT_ARUCO_ORIGINAL",
    help="type of ArUCo tag to detect")
args = vars(ap.parse_args())

# define names of each possible ArUco tag OpenCV supports
ARUCO_DICT = {
    # "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    # "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    # "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    # "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    # "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    # "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    # "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    # "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    # "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    # "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    # "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    # "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    # "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    # "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    # "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    # "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL

}

# verify that the supplied ArUCo tag exists and is supported by
# OpenCV
if ARUCO_DICT.get(args["type"], None) is None:
    print("[INFO] ArUCo tag of '{}' is not supported".format(
        args["type"]))
    sys.exit(0)

# load the ArUCo dictionary and grab the ArUCo parameters
print("[INFO] detecting '{}' tags...".format(args["type"]))
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters_create()

# arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]])
# arucoParams =  cv2.aruco.DetectorParameters()
# detector = cv.aruco.ArucoDetector(dictionary, parameters)

# initialize the video stream and allow the camera sensor to warm up
# print("[INFO] starting video stream...")
# vs = VideoStream(src=0).start()
# time.sleep(2.0)

# load the input image from disk and resize it
# print("[INFO] loading image...")
# frame = cv2.imread("aruco.png")
# frame = imutils.resize(frame, width=600)

output_csv_file = 'aruco_coords.csv'
fields = ['X', 'Y']
with open(output_csv_file, 'w', newline='\n') as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()

with open(output_csv_file, 'a', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    
    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 600 pixels
        # frame = vs.read()
        frames = pipeline.wait_for_frames()
        frame = frames.get_color_frame()
        # frame = imutils.resize(frame, width=1000)
        # breakpoint()
        # cv2.imwrite("orig.png", frame)
        frame = np.asanyarray(frame.get_data()) #480 by #640
        # print(frame.shape) 
        # detect ArUco markers in the input frame
        (corners, ids, rejected) = cv2.aruco.detectMarkers(frame,
            arucoDict, parameters=arucoParams)

        # verify *at least* one ArUco marker was detected
        if len(corners) > 0:
            # flatten the ArUco IDs list
            ids = ids.flatten()
            print(len(corners))
            # lenc = len(corners)

            
            # loop over the detected ArUCo corners
            for (markerCorner, markerID) in zip(corners, ids):
                # extract the marker corners (which are always returned
                # in top-left, top-right, bottom-right, and bottom-left
                # order)
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners

                # convert each of the (x, y)-coordinate pairs to integers
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))

                # draw the bounding box of the ArUCo detection
                cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)

                # compute and draw the center (x, y)-coordinates of the
                # ArUco marker
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)			
                print(cX, cY)
        
        # test = 1
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writerow({'X': cX, 'Y': cY})
                # writer = csv.writer(file)
                # writer.writerows([cX, cY])
                

                
                


            # draw the ArUco marker ID on the frame
            cv2.putText(frame, str(markerID),
                (topLeft[0], topLeft[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        # if(lenc== 14):
        # 		cv2.imwrite("all_marks.png", frame)
        # 		break
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # 
    cv2.destroyAllWindows()
    # vs.stop()
