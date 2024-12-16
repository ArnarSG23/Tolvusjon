import cv2
import time #for fps
from ultralytics import YOLO
from datetime import datetime, timedelta #manage and calc intervals between recipe generation
from collections import defaultdict #dictionary subclass for missing keys
import google.generativeai as genai
from threading import Thread, Lock #for smooth real-time detection and querying
import numpy as np  

# Load the YOLO model
model = YOLO('best_11_2.pt')

# Configure the Generative AI API
genai.configure(api_key="AIzaSyDD9nEQpmspR31wPVYDrwtjTJRDAIzdJzY")
genai_model = genai.GenerativeModel("gemini-1.5-flash")

# Global variables
next_check_time = datetime.now() + timedelta(seconds=10) #time between recipe generations
startMSG = '''Create a recipe that uses the ingredients the user sends. 
You don't need to use all the ingredients and can only add spices, cooking oil, water, and nothing else. 
Here are the ingredients: '''
detecteds = defaultdict(list) #tracks detected ingredients and the bounding boxes within in frame
stable_ingredients = defaultdict(float) #accumulate ingredient quantity for recipes every 10sec
lock = Lock()
frame_lock = Lock()
recipe_lock = Lock()
cap = cv2.VideoCapture(0)
prev_time = time.time() #initialize time for FPS calculation
annotated_frame = None
last_ingredients_display = None
recipe_text = None

# Define units based on ingredient type
unit_mapping = {
    "Bacon": "grams",
    "banana": "pieces",
    "Bread": "pieces",
    "Canned Tomatoes": "grams",
    "Chicken": "grams",
    "Corn": "grams",
    "Fish": "grams",
    "Flour": "grams",
    "Ground beef": "grams",
    "Lettuce": "grams",
    "Meat": "grams",
    "Milk": "liters",
    "Mushroom": "pieces",
    "Paprika": "pieces",
    "Parmesan": "grams",
    "Pasta": "grams",
    "Potato": "pieces",
    "Rice": "grams",
    "Sugar": "grams",
    "apple": "pieces",
    "butter": "grams",
    "cheese": "grams",
    "egg": "pieces",
    "garlic": "pieces",
    "onion": "pieces",
    "tomato": "pieces",
}

density_factors = {
    "Bacon": 1.9,
    "Canned Tomatoes": 1.05,
    "Chicken": 16.05,
    "Corn": 1.6,
    "Fish": 1.1,
    "Flour": 0.6,
    "Ground beef": 15.1,
    "Lettuce": 0.2,
    "Meat": 16.5,
    "Milk": 1.03,
    "Parmesan": 17.2,
    "Pasta": 1.95,
    "Rice": 1.85,
    "Sugar": 7.6,
    "butter": 13.9,
    "cheese": 15.0,
}


# Check if two bounding boxes are duplicates based on position
def is_duplicate(box1, box2, threshold=50):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Check if the centers of the boxes are within the threshold distance
    center1 = (x1 + w1 // 2, y1 + h1 // 2)
    center2 = (x2 + w2 // 2, y2 + h2 // 2)
    distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
    return distance < threshold

def detect():
    global detecteds, prev_time, annotated_frame, last_ingredients_display

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            continue

        # Run YOLO model
        results = model(frame, verbose=False)
        annotated = results[0].plot()

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        cv2.putText(annotated, f"FPS: {fps:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Process detection
        current_detecteds = defaultdict(list)
        for box in results[0].boxes: #go through the list of objects detected in frame
            class_id = int(box.cls)
            ingredient_name = results[0].names[class_id] #map class id to ingredient name
            bounding_box = box.xyxy[0].cpu().numpy() #get bounding box coordinates
            # Converting to pixel coord, x and y = top left corner of bounding box, w = width of box, h = height of box
            x, y, w, h = int(bounding_box[0]), int(bounding_box[1]), int(bounding_box[2] - bounding_box[0]), int(bounding_box[3] - bounding_box[1])

            # Check if this detection is a duplicate
            if not any(is_duplicate((x, y, w, h), existing_box) for existing_box in detecteds[ingredient_name]):
                current_detecteds[ingredient_name].append((x, y, w, h))

        # Update global detection state using Lock for thread-safe access
        with lock:
            for ingredient, boxes in current_detecteds.items():
                detecteds[ingredient].extend(boxes)

        # Update ingredient list for display
        new_ingredients_display = []
        for ingredient, boxes in detecteds.items():
            unit = unit_mapping.get(ingredient, "units")
            density = density_factors.get(ingredient, 1.0)
            quantity = len(boxes) if unit == "pieces" else \
                sum((w * h) / 10000 * density for _, _, w, h in boxes) #reference area is 10.000pixels for 100gr
            new_ingredients_display.append(f"{ingredient} ({round(quantity, 2)} {unit})")

        last_ingredients_display = new_ingredients_display

        # Render the ingredient list
        y_offset = 50
        for ingredient_text in last_ingredients_display:
            cv2.putText(
                annotated,
                ingredient_text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            y_offset += 20

        with frame_lock:
            annotated_frame = annotated

def query():
    global next_check_time, detecteds, stable_ingredients, recipe_text
    # Check if its time to query AI
    if datetime.now() < next_check_time:
        return
    next_check_time = datetime.now() + timedelta(seconds=10)

    # Process detections and calculate stable quantities
    total_quantities = defaultdict(float) #dictonary that stores aggregated quantities of detected ingredients

    with lock: #for safe access to shared gloabal variables
        for ingredient, boxes in detecteds.items():
            unit = unit_mapping.get(ingredient, "units")
            density = density_factors.get(ingredient, 1.0)
            quantity = len(boxes) if unit == "pieces" else \
                sum((w * h) / 10000 * density for _, _, w, h in boxes)
            total_quantities[ingredient] += quantity
        detecteds.clear() #reset detection state after processing to avoid duplicate processing in next query

    # Update stable ingredients by accumulating quantites of ingredients across frames
    for ingredient, quantity in total_quantities.items():
        stable_ingredients[ingredient] += quantity

    # Build a query string
    ingredients = ", ".join([f"{ingredient} ({round(quantity, 2)} {unit_mapping.get(ingredient, 'units')})"
                             for ingredient, quantity in stable_ingredients.items()])
    # Query the API
    if ingredients:
        print(f"Detected ingredients: {ingredients}")
        msg = f"{startMSG} {ingredients}"
        try:
            response = genai_model.generate_content(msg)
            with recipe_lock:
                recipe_text = response.text
                stable_ingredients.clear() #clear after successful query
        except Exception as e:
            print(f"Error querying the API: {e}")


def display_recipe_window(recipe_text):

    # Split the recipe text into lines
    lines = recipe_text.split("\n")

    line_height = 20  
    margin = 10 
    width = 600  
    height = max(400, len(lines) * line_height + 2 * margin) #minimum height of 400

    recipe_canvas = np.ones((height, width, 3), dtype="uint8") * 255

    y_offset = margin + line_height #start after the top margin
    for line in lines:
        cv2.putText(
            recipe_canvas,
            line,
            (margin, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,  #reduced font size
            (0, 0, 0),  #black text
            1,
            lineType=cv2.LINE_AA,
        )
        y_offset += line_height  #increment y position for the next line

    # Display the recipe window
    cv2.imshow("Generated Recipe", recipe_canvas)

if __name__ == "__main__":
    try:
        detect_thread = Thread(target=detect)
        detect_thread.daemon = True
        detect_thread.start()

        while True:
            with frame_lock:
                if annotated_frame is not None:
                    cv2.imshow("Object Detection", annotated_frame)

            if recipe_text:
                print(f"Generated Recipe: {recipe_text}")
                with recipe_lock:
                    display_recipe_window(recipe_text)  #display recipe
                    recipe_text = None  #clear after displaying

            query_thread = Thread(target=query)
            query_thread.start()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
