import streamlit as st
import numpy as np
import cv2
import tempfile
import requests
from PIL import Image

st.title("Gemini Vision Annotator")

# Upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Convert image to OpenCV format
    image = np.array(image)
    
    st.write("Draw bounding boxes by entering coordinates:")
    annotations = []
    
    num_boxes = st.number_input("Number of bounding boxes", min_value=1, step=1, value=1)
    for i in range(num_boxes):
        x = st.number_input(f"Box {i+1} - X", min_value=0, value=0)
        y = st.number_input(f"Box {i+1} - Y", min_value=0, value=0)
        width = st.number_input(f"Box {i+1} - Width", min_value=1, value=50)
        height = st.number_input(f"Box {i+1} - Height", min_value=1, value=50)
        annotations.append({"x": x, "y": y, "width": width, "height": height})
    
    # Draw annotations on the image
    for ann in annotations:
        cv2.rectangle(image, (ann["x"], ann["y"]), (ann["x"] + ann["width"], ann["y"] + ann["height"]), (255, 0, 0), 2)
    
    st.image(image, caption="Annotated Image", use_column_width=True)
    
    # Save and send to API
    if st.button("Submit to Gemini Vision"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            annotated_image = Image.fromarray(image)
            annotated_image.save(temp_file.name)
            temp_file_path = temp_file.name
        
        # Define the files variable
        files = {"file": open(temp_file_path, "rb")}
        
        # Example API call (replace with actual Gemini Vision API endpoint)
        url = "https://generativelanguage.googleapis.com/v1/models/gemini-pro-vision:generateContent?key=AIzaSyBex0XFQU5NVjft-VP1tUlT4JYojjvA9aA"


        try:
            response = requests.post(url, files=files, data={"annotations": str(annotations)})
            response.raise_for_status()
            st.success("Successfully submitted to Gemini Vision API!")
            st.json(response.json())
        except requests.exceptions.ConnectionError:
            st.error("Failed to connect to API. Ensure the server is running.")
        except requests.exceptions.RequestException as e:
            st.error(f"API error: {e}")

