import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load pre-trained car color detection model
model = load_model(r'path_of_file\COLOR_detection_model.keras')

# List of color names corresponding to predicted numbers (indices)
color_names = ['beige', 'black', 'blue', 'brown', 'gold', 'green', 'grey', 'orange', 'pink', 'purple', 'red', 'silver', 'tan', 'white', 'yellow']

# Load pre-trained object detection model (MobileNet SSD)
net = cv2.dnn.readNetFromCaffe(r"path_of_file\deploy.prototxt",
                               r"path_of_file\mobilenet_iter_73000.caffemodel")

current_zoom = 1.0  # Initialize current_zoom
img_tk = None  # Global variable to store the image for zooming
pil_img = None  # Global variable to store the original PIL image for zooming

# Function to detect car color, draw rectangles, show color name, and display car count
def detect_and_draw(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Prepare the image for object detection
    blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    car_count = 0  # Car counter
    people_count = 0  # People counter

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:  # Confidence threshold
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
            (x, y, x1, y1) = box.astype("int")
            w, h = x1 - x, y1 - y

            # Ensure the bounding box is within the image dimensions
            if x + w > img_rgb.shape[1] or y + h > img_rgb.shape[0]:
                continue  # Skip this bounding box if it's out of bounds

            if idx == 7:  # Class ID for car in COCO dataset
                # Crop the image based on the bounding box
                cropped_img = img_rgb[y:y + h, x:x + w]

                # Check if the cropped image is valid (non-empty)
                if cropped_img.size == 0:
                    print(f"Invalid crop at coordinates: {(x, y, w, h)}")
                    continue

                # Resize the cropped image
                resized_img = cv2.resize(cropped_img, (128, 128))

                # Convert image for model prediction
                image_array = img_to_array(resized_img) / 255.0
                image_array = np.expand_dims(image_array, axis=0)

                # Predict car color
                prediction = model.predict(image_array)
                predicted_index = np.argmax(prediction)  # Get the index of the predicted class

                # Map predicted index to color name
                predicted_color = color_names[predicted_index]

                # Draw rectangles based on predicted color
                if predicted_color == 'blue':
                    cv2.rectangle(image, (x, y), (x1, y1), (0, 0, 255), 3)  # Red rectangle for blue cars
                else:
                    cv2.rectangle(image, (x, y), (x1, y1), (255, 0, 0), 3)  # Blue rectangle for other cars

                # Put the color name on top of the rectangle
                cv2.putText(image, predicted_color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                car_count += 1  # Increment car count

            elif idx == 15:  # Class ID for person in COCO dataset
                cv2.rectangle(image, (x, y), (x1, y1), (0, 255, 0), 3)  # Green rectangle for people
                cv2.putText(image, 'Person', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                people_count += 1  # Increment people count

    return image, car_count, people_count

# Function to open and display an image, fitting it to the full screen
def open_image():
    global pil_img, img_tk
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png")])
    if file_path:
        image = cv2.imread(file_path)
        processed_img, car_count, people_count = detect_and_draw(image)
        
        # Get screen size
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight() - 50  # Leave space for buttons at the bottom

        # Resize image to fit the screen
        processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(processed_img_rgb)  # Store original PIL image for zooming
        pil_img = pil_img.resize((screen_width, screen_height), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(pil_img)
        
        # Update image label
        image_label.config(image=img_tk)
        image_label.image = img_tk

        # Display car and people count
        print(f'Total Car Count: {car_count}')
        print(f'Total People Count: {people_count}')

# Function to zoom in and out using the mouse wheel
def zoom(event):
    global pil_img, img_tk, current_zoom
    if pil_img is None:
        return  # No image loaded

    # Zoom in or out based on mouse wheel direction
    scale_factor = 1.1 if event.delta > 0 else 0.9
    current_zoom *= scale_factor

    # Calculate new size
    width, height = pil_img.size
    new_size = (int(width * current_zoom), int(height * current_zoom))

    # Resize the image
    resized_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
    img_tk = ImageTk.PhotoImage(resized_img)

    # Update the label with the resized image
    image_label.config(image=img_tk)
    image_label.image = img_tk

# Function to open and process a video
def open_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
    if file_path:
        cap = cv2.VideoCapture(file_path)
        total_car_count = 0
        total_people_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, car_count, people_count = detect_and_draw(frame)
            total_car_count += car_count
            total_people_count += people_count

            cv2.imshow('Video', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f'Total Car Count: {total_car_count}')
        print(f'Total People Count: {total_people_count}')


# GUI Setup
root = tk.Tk()
root.attributes("-fullscreen", True)
root.title("Car Color Detection")

# Set window to full screen
root.attributes('-fullscreen', True)

# Exit full screen with 'Esc' key
root.bind("<Escape>", lambda event: root.attributes("-fullscreen", False))

# Bind mouse wheel to zoom function
root.bind("<MouseWheel>", zoom)
# Image display label
image_label = tk.Label(root)
image_label.pack(fill=tk.BOTH, expand=True)  # Fill all available space

# Buttons Frame (bottom)
buttons_frame = tk.Frame(root)
buttons_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

# Open Image Button
open_button = tk.Button(buttons_frame, text="Open Image", command=open_image)
open_button.pack(side=tk.LEFT, padx=20)

# Open Video Button
open_video_button = tk.Button(buttons_frame, text="Open Video", command=open_video)
open_video_button.pack(side=tk.LEFT, padx=20)

# Run the application
root.mainloop()
