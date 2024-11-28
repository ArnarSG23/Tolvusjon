import cv2
import numpy as np
import time

def draw_extended_line(frame, rho, theta):
    """Draw a line that spans the entire frame."""
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a * rho, b * rho

    #Calculate the intersection points with the frame boundaries
    #Line equation: x = x0 + t * (-b), y = y0 + t * a
    height, width = frame.shape[:2]
    
    #Points at frame boundaries
    points = []
    if b != 0:  #Avoid division by zero
        points.append((0, int(rho / b)))  
        points.append((width, int((rho - width * a) / b))) 
    if a != 0:  #Avoid division by zero
        points.append((int(rho / a), 0))  
        points.append((int((rho - height * b) / a), height)) 

    #Filter points within frame boundaries
    valid_points = [
        (int(x), int(y))
        for x, y in points
        if 0 <= x <= width and 0 <= y <= height
    ]

    #If we have two valid points, draw the line
    if len(valid_points) == 2:
        cv2.line(frame, valid_points[0], valid_points[1], (0, 255, 0), 2) 


# 0 = phone camera
# 1 = computer camera
cap = cv2.VideoCapture(0)

prev_time = time.time() #Initialize time for FPS

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    

    #Gaussian filter - less noise and smoother image
    blurred = cv2.GaussianBlur(gray, (5, 5), 1) #kernel matrix and sigma
    
    #Canny edge detection - lower and upper threshold
    edges = cv2.Canny(blurred, 130, 150)
    
    #Hough Line Transform to detect lines
    lines = cv2.HoughLines(edges, 2, np.pi / 180, 200)  

    #Draw the detected lines on the original frame
    if lines is not None:
        # Limit to top 4 most prominent lines
        for i in range(min(len(lines), 4)):
            rho, theta = lines[i][0]
            draw_extended_line(frame, rho, theta)

    #Side by side frames
    cv2.putText(frame, f"FPS: {fps:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    combined = np.hstack((gray, edges))  
    #cv2.imshow('Gray and Edges', combined)
    cv2.imshow('Frame with Hough Lines', frame)

    #cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
