import cv2
import numpy as np
import time

def find_intersection(line1, line2):
    """Find intersection point of two lines given in (rho, theta) format."""
    rho1, theta1 = line1
    rho2, theta2 = line2

    #Line coefficients
    a1, b1 = np.cos(theta1), np.sin(theta1)
    a2, b2 = np.cos(theta2), np.sin(theta2)

    #Compute determinant to check for parallel lines
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-10: 
        return None

    # Compute intersection point
    x = (b2 * rho1 - b1 * rho2) / det
    y = (a1 * rho2 - a2 * rho1) / det
    return int(x), int(y)

def remove_duplicate_points(points, threshold=20):
    """Remove duplicate or near-duplicate points."""
    filtered_points = []
    for point in points:
        if all(np.linalg.norm(np.array(point) - np.array(p)) > threshold for p in filtered_points):
            filtered_points.append(point)
    return filtered_points

def define_corners(points):
    """Sort points to identify the corners of the quadrangle."""
    # Sort points by y-coordinate (top to bottom), and then x-coordinate (left to right)
    points = sorted(points, key=lambda p: (p[1], p[0]))

    # Top-left and top-right
    top_left, top_right = sorted(points[:2], key=lambda p: p[0])
    # Bottom-left and bottom-right
    bottom_left, bottom_right = sorted(points[2:], key=lambda p: p[0])

    return [top_left, top_right, bottom_right, bottom_left]


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
    #edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.Canny(blurred, 130, 150)
    
    #Hough Line Transform to detect lines
    lines = cv2.HoughLines(edges, 2, np.pi / 180, 300)  

    if lines is not None:
        lines = lines[:4]

        intersections = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                point = find_intersection(lines[i][0], lines[j][0])
                if point:
                    x, y = point
                    if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]: 
                        intersections.append((x, y))

        #Remove duplicate or near-duplicate points
        unique_intersections = remove_duplicate_points(intersections)

        #Identify and draw the corners
        if len(unique_intersections) >= 4:
            corners = define_corners(unique_intersections[:4])
            for point in corners:
                cv2.circle(frame, point, 10, (0, 255, 0), -1)  # Green circles

            #Perform perspective transformation
            #Define the destination points (rectangular)
            width, height = 400, 300  #Output image dimensions
            dst_points = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

            #Compute the perspective transform matrix
            src_points = np.array(corners, dtype=np.float32)
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)

            #Warp the perspective to get a top-down view
            warped = cv2.warpPerspective(frame, matrix, (width, height))

            #Show the warped image
            cv2.imshow('Warped Perspective', warped)

    cv2.putText(frame, f"FPS: {fps:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    combined = np.hstack((gray, edges))  
    #cv2.imshow('Gray and Edges', combined)
    # Show frame with intersections
    cv2.imshow('Frame with Intersections', frame)

    #cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



