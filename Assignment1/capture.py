# ------------------------------------ Code from Torfi -----------------------------------------
# import cv2

# # 0 = phone camera
# # 1 = computer camera
# cap = cv2.VideoCapture(0)

# while(True):
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# ------------------------------------ Adding FPS -----------------------------------------
# import cv2
# import time

# # 0 = phone camera
# # 1 = computer camera
# cap = cv2.VideoCapture(1)

# prev_time = time.time() #Initialize time for FPS


# while(True):
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     #Calculate FPS
#     current_time = time.time()
#     fps = 1 / (current_time - prev_time)
#     prev_time = current_time

#     #Displaying FPS on frame 
#     cv2.putText(frame, f"FPS: {fps:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     cv2.imshow('frame',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# ------------------------------------ Adding Brightest -----------------------------------------
# import cv2
# import time

# # 0 = phone camera
# # 1 = computer camera
# cap = cv2.VideoCapture(1)

# prev_time = time.time() #Initialize time for FPS


# while(True):
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     #Calculate FPS
#     current_time = time.time()
#     fps = 1 / (current_time - prev_time)
#     prev_time = current_time
#     print(f"FPS: {fps:.0f}") #print to terminal

#     #Finding and marking the brightest spot
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)
#     cv2.circle(frame, max_loc, 15, (255, 0, 0), 2)  #Blue circle

#     #Displaying FPS on frame 
#     cv2.putText(frame, f"FPS: {fps:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     cv2.imshow('frame',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# ------------------------------------ Adding Reddest -----------------------------------------
import cv2
import time
import numpy as np

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

    #print(f"FPS: {fps:.2f}")#print to terminal

    #Finding and marking the brightest spot
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)
    cv2.circle(frame, max_loc, 15, (255, 0, 0), 2)  #Blue circle

    #Finding and marking the reddest spot - convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #Define red color ranges in HSV
    lower_red1 = (0, 50, 50)
    upper_red1 = (10, 255, 255)
    lower_red2 = (170, 50, 50)
    upper_red2 = (180, 255, 255)
    #Mask for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2

    #Weighted redness for pixels in the red mask
    redness = red_mask * (hsv[:, :, 1].astype(np.float32) * hsv[:, :, 2].astype(np.float32))

    #Find the location of the maximum red intensity
    _, max_val_red, _, max_loc_red = cv2.minMaxLoc(redness)
    cv2.circle(frame, max_loc_red, 15, (0, 0, 255), 2)  # Red circle for reddest spot

    #cv2.imshow("Red Mask", red_mask)

    #Displaying FPS on frame 
    cv2.putText(frame, f"FPS: {fps:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ------------------------------------ For-loop instead of functions -----------------------------------------

# import cv2
# import time
# import numpy as np

# # 0 = phone camera
# # 1 = computer camera
# cap = cv2.VideoCapture(1)

# prev_time = time.time() #Initialize time for FPS


# while(True):
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     #Calculate FPS
#     current_time = time.time()
#     fps = 1 / (current_time - prev_time)
#     prev_time = current_time

#     #print(f"FPS: {fps:.2f}")#print to terminal

#     #Finding and marking the brightest spot - for-loop
#     max_val = 0
#     max_loc = (0, 0)
#     for i in range(gray.shape[1]):  # Loop over columns 
#         for j in range(gray.shape[0]):  # Loop over rows 
#             if gray[j, i] > max_val:
#                 max_val = gray[j, i]
#                 max_loc = (i, j)

#     cv2.circle(frame, max_loc, 15, (255, 0, 0), 2)  #Blue circle

#     #Finding and marking the reddest spot - convert to HSV
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     #Define red color ranges in HSV
#     lower_red1 = (0, 50, 50)
#     upper_red1 = (10, 255, 255)
#     lower_red2 = (170, 50, 50)
#     upper_red2 = (180, 255, 255)
#     #Mask for red color
#     mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
#     mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
#     red_mask = mask1 + mask2

#     #Weighted redness for pixels in the red mask
#     redness = red_mask * (hsv[:, :, 1].astype(np.float32) * hsv[:, :, 2].astype(np.float32))

#     #Find the location of the maximum red intensity
#     _, max_val_red, _, max_loc_red = cv2.minMaxLoc(redness)
#     cv2.circle(frame, max_loc_red, 15, (0, 0, 255), 2)  # Red circle for reddest spot

#     #cv2.imshow("Red Mask", red_mask)

#     #Displaying FPS on frame - cv2.putText(image, text, org, fontFace, fontScale, color, thickness, lineType)
#     cv2.putText(frame, f"FPS: {fps:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     cv2.imshow('frame',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()