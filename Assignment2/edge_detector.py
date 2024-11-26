#----------------------------------------------------------- Canny Edge Detection -----------------------------------------------------------------------
# import cv2
# import numpy as np

# # 0 = phone camera
# # 1 = computer camera
# cap = cv2.VideoCapture(0)

# while(True):
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     #Gaussian filter - less noise and smoother image
#     blurred = cv2.GaussianBlur(gray, (5, 5), 1) #kernel matrix and sigma
    
#     #Canny edge detection - lower and upper threshold
#     edges = cv2.Canny(blurred, 50, 150)
    
#     #Side by side frames
#     combined = np.hstack((gray, edges))  
#     cv2.imshow('Gray and Edges', combined)

#     #cv2.imshow('frame',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

#----------------------------------------------------------- Coordinates of edge pixels written into an array -----------------------------------------------------------------------
# import cv2
# import numpy as np

# # 0 = phone camera
# # 1 = computer camera
# cap = cv2.VideoCapture(0)

# while(True):
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     #Gaussian filter - less noise and smoother image
#     blurred = cv2.GaussianBlur(gray, (5, 5), 1) #kernel matrix and sigma
    
#     #Canny edge detection - lower and upper threshold
#     edges = cv2.Canny(blurred, 50, 150)
    
#     #Coordinates of edge pixels
#     coordinates = np.column_stack(np.where(edges > 0))  #all non zero values are edges
    
#     #Side by side frames
#     combined = np.hstack((gray, edges))  
#     cv2.imshow('Gray and Edges', combined)

#     #cv2.imshow('frame',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

#----------------------------------------------------------- RANSAC to fit a straight line -----------------------------------------------------------------------
import cv2
import numpy as np
import random
import time

def RANSAC(points, p, e, threshold):
    
    line = None
    best_inliers = 0
    num_iterations = int(np.log(1 - p) / np.log(1 - (1 - e)**2))

    for _ in range(num_iterations):
        #Random two points
        sample = random.sample(list(points), 2)
        x1, y1 = sample[0]
        x2, y2 = sample[1]

        #Line parameters
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2

        #Line parameters normalized
        norm = np.sqrt(a**2 + b**2)
        a, b, c = a / norm, b / norm, c / norm

        #Distnce from points to the line
        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c)

        #Counting inliers
        inliers = np.sum(distances < threshold)

        #Kkep the line with most inliers
        if inliers > best_inliers:
            line = (a, b, c)
            best_inliers = inliers

    return line

# 0 = phone camera
# 1 = computer camera
cap = cv2.VideoCapture(1)

prev_time = time.time() #Initialize time for FPS

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    #Gaussian filter - less noise and smoother image
    blurred = cv2.GaussianBlur(gray, (5, 5), 3) #kernel matrix and sigma
    
    #Canny edge detection - lower and upper threshold
    edges = cv2.Canny(blurred, 50, 150)
    
    #Coordinates of edge pixels
    coordinates = np.column_stack(np.where(edges > 0))  #all non zero values are edges
    coordinates = coordinates[:, [1, 0]]  # Convert (y, x) to (x, y)

    if len(coordinates) > 2: 
        
        line = RANSAC(coordinates, 0.75, 0.85, 3)
        a, b, c = line

        #Endpoints
        height, width = frame.shape[:2]
        if b != 0:
            x1, y1 = 0, int(-c / b)  # Line intersects the left edge
            x2, y2 = width, int((-c - a * width) / b)  # Line intersects the right edge

            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  
    
    #Side by side frames
    combined = np.hstack((gray, edges))  
    #cv2.imshow('Gray and Edges', combined)
    cv2.putText(frame, f"FPS: {fps:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("RANSAC", frame)

    #cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
