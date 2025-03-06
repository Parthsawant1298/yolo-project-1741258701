import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import os

def main():
    st.title('Object Detection with YOLOv8')
    st.write('Upload an image to detect objects')
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Save the uploaded image temporarily
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Perform object detection
        if st.button('Detect Objects'):
            with st.spinner('Detecting...'):
                # Load the model
                model = YOLO("best_model.pt")
                
                # Run inference
                results = model.predict(source="temp_image.jpg", conf=0.25)
                
                # Get the first result (we only have one image)
                result = results[0]
                
                # Display result
                result_img = result.plot()
                result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                st.image(result_img_rgb, caption='Detection Result', use_column_width=True)
                
                # Display detection details
                st.subheader('Detection Results:')
                
                # Get class names dictionary
                names = model.names
                
                # Display each detection
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    class_name = names[cls_id]
                    confidence = box.conf[0].item()
                    coordinates = box.xyxy[0].tolist()  # get box coordinates in (top, left, bottom, right) format
                    
                    st.write(f"**Class:** {class_name}, **Confidence:** {confidence:.2f}")
                    st.write(f"**Coordinates:** [x1={coordinates[0]:.1f}, y1={coordinates[1]:.1f}, x2={coordinates[2]:.1f}, y2={coordinates[3]:.1f}]")
                    st.write("---")
                
                # Clean up
                try:
                    os.remove("temp_image.jpg")
                except:
                    pass

if __name__ == "__main__":
    main()