'''
#======= DESCRIPTION =======

This is only a test program for stop and lane line detection so a local image will be used as the input.
Normal method for extracting lane lines using canny edge detection and hough transform is used. 
The roadlines that are not inside the drivable surface based on segmentation are removed from the ROI. 
The stop lines and lane lines are then separated into different colours. 

To Use:
1) Set fileName to the name of the image being transformed.
2) Ensure the image and this file are in the same folder.
3) Run the program.

'''

import cv2
import numpy as np
import math

mean_height = 0
mean_width = 0


#======= Helper Functions =======

def selectROI(edges):
	height, width = edges.shape
	mask = np.zeros_like(edges)

	polygon = np.array([[
        (0, height * 3 / 16),  #(0, height * 9 / 16), 
        (width, height * 6 / 16), #(width, height * 9 / 16),
        (width, height),
        (0, height),
    	]], np.int32)

    	cv2.fillPoly(mask, polygon, 255)
    	cropped_edges = cv2.bitwise_and(edges, mask)

	return cropped_edges



#Separate lane into lane and stop lane 
#If a detected lane is orthorgonal to stop lane, then 
#Check if image is loaded fine
#we find edges
#======= Main =======


frame = cv2.imread('/home/celene/git_workspace/Stop-Line-Detection-and-Tracking/newINPUT.png')
scale_percent = 25 # percent of original size
width = int(frame.shape[1] * scale_percent / 100)
height = int(frame.shape[0] * scale_percent / 100)
dim = (width, height)

frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
#hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#Instead of filtering for the yellow lanes
#Better to filter out the black background

lower_lane = np.array([20, 190, 250])
upper_lane = np.array([40, 220, 255])
mask = cv2.inRange(frame, lower_lane, upper_lane)
#edges = cv2.Canny(mask, 100, 400)

#focus_edges = selectROI(edges)
lines = cv2.HoughLinesP(mask,1,np.pi/180,80,minLineLength=100,maxLineGap=30)

#Detecting Edges
blocksize = 2
aperturesize = 3
corner = cv2.cornerHarris(mask, blocksize, aperturesize, 0.04)


#Normalizing
corner_norm = np.empty(corner.shape, dtype = np.float32)
cv2.normalize(corner, corner_norm, alpha = 0, beta = 255, norm_type=cv2.NORM_MINMAX)
corner_norm_scaled = cv2.convertScaleAbs(corner_norm)




try:#for real time in case no lines detected
	for line in lines:
		x1,y1,x2,y2 = line[0]
		theta = line[0][1]-360
		print(theta)
		if theta > 150 or theta < 50:
		#if theta > np.pi/180 * 170 or theta < np.pi/180 * 10: #vertical lines
			cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)
		elif theta > 10 and theta < 170: 
		#elif theta > np.pi/180 * 20 and theta < np.pi/180 * 160: #horizontal lines
			cv2.line(frame,(x1,y1),(x2,y2),(255,0,0),2)
except: 
	pass


#Drawing circle around edges
for i in range(corner_norm.shape[0]):
	for j in range(corner_norm_scaled.shape[1]):
		if int(corner_norm[i,j]) > 200:
			#cv2.circle(corner_norm_scaled,(j,i),5,(0),2)
			cv2.circle(corner_norm_scaled,(j,i), 15, (0,0,255), 2)
			cv2.circle(frame,(j,i), 15, (0,0,255), 2)
# Separate into stop lines and lane lines.



# Dashed Lines 



#Put this into local directory instead of 
cv2.imwrite('/home/celene/git_workspace/Stop-Line-Detection-and-Tracking/result.jpg',frame)
#cv2.imshow('hsv', hsv)
cv2.imshow('mask', mask)
#cv2.imshow('canny', edges)
cv2.imshow('output', frame)
cv2.imshow('corners', corner_norm_scaled)
cv2.waitKey(50000)
#cv2.imwrite('/home/celene/git_workspace/Stop-Line-Detection-and-Tracking/output.jpg',frame)


print("Program Ends")


