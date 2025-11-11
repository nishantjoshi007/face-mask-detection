import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import json
import os
import time

# Page configuration
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .upload-box {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 40px;
        text-align: center;
        background-color: white;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model and class indices
@st.cache_resource
def load_model_and_classes():
    """Load the trained model and class indices"""
    try:
        # Try loading different model formats
        if os.path.exists('models/face_mask_detector.keras'):
            model = keras.models.load_model('models/face_mask_detector.keras')
        elif os.path.exists('models/best_mask_detector.h5'):
            model = keras.models.load_model('models/best_mask_detector.h5')
        elif os.path.exists('models/face_mask_detector.h5'):
            model = keras.models.load_model('models/face_mask_detector.h5')
        else:
            st.error("No model file found in 'models/' folder")
            return None, None
        
        # Load class indices
        with open('models/class_indices.json', 'r') as f:
            class_indices = json.load(f)
        
        # Reverse mapping
        class_labels = {v: k for k, v in class_indices.items()}
        
        return model, class_labels
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Load face detector
@st.cache_resource
def load_face_detector():
    """Load Haar Cascade face detector"""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    return face_cascade

# Preprocess image for model
def preprocess_face(face_img, target_size=(224, 224)):
    """Preprocess face image for model prediction"""
    face_img = cv2.resize(face_img, target_size)
    face_img = face_img / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    return face_img

# Detect faces and predict
def detect_and_predict(image, model, class_labels, face_cascade):
    """Detect faces and predict mask status"""
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(60, 60)
    )
    
    results = []
    
    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = img_rgb[y:y+h, x:x+w]
        face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        
        # Preprocess and predict
        preprocessed_face = preprocess_face(face_roi_rgb)
        prediction = model.predict(preprocessed_face, verbose=0)
        
        class_idx = np.argmax(prediction)
        confidence = prediction[0][class_idx] * 100
        label = class_labels[class_idx]
        
        # Draw rectangle and label
        color = (0, 255, 0) if label == 'with_mask' else (0, 0, 255)
        cv2.rectangle(img_rgb, (x, y), (x+w, y+h), color, 3)
        
        # Add label with background
        label_text = f"{label.replace('_', ' ').title()}: {confidence:.1f}%"
        
        # Calculate text size for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
        
        # Draw background rectangle for text
        cv2.rectangle(img_rgb, (x, y - text_height - 10), (x + text_width, y), color, -1)
        
        # Put text
        cv2.putText(img_rgb, label_text, (x, y - 5), 
                    font, font_scale, (255, 255, 255), thickness)
        
        results.append({
            'label': label,
            'confidence': confidence,
            'bbox': (x, y, w, h)
        })
    
    # Convert back to RGB for display
    output_image = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    
    return output_image, results

# Main app
def main():
    # Load model and face detector
    model, class_labels = load_model_and_classes()
    face_cascade = load_face_detector()
    
    if model is None:
        st.error(" Failed to load model. Please check the model files in the 'models/' folder.")
        st.stop()
    
    # Title and description
    st.title(" Face Mask Detection System")
    st.markdown("""
    ### Welcome to the AI-Powered Face Mask Detection App!
    Upload an image or capture from webcam to detect face masks in real-time.
    """)
    
    # Sidebar
    st.sidebar.title(" Settings")
    st.sidebar.markdown("---")
    
    # Detection mode
    detection_mode = st.sidebar.selectbox(
        "Choose Detection Mode:",
        [" Upload Image", " Capture from Webcam", " Batch Processing"]
    )
    
    st.sidebar.markdown("---")
    
    # Model info
    with st.sidebar.expander("ℹ Model Information"):
        st.write("**Architecture:** MobileNetV2")
        st.write("**Classes:** With Mask, Without Mask")
        st.write("**Input Size:** 224x224")
        st.write("**Accuracy:** ~98%+")
    
    # About
    with st.sidebar.expander(" About"):
        st.write("""
        This application uses deep learning to detect whether people 
        are wearing face masks correctly. Built with:
        - TensorFlow/Keras
        - OpenCV
        - Streamlit
        """)
    
    # Main content area
    st.markdown("---")
    
    # Mode 1: Upload Image
    if detection_mode == " Upload Image":
        st.subheader(" Upload an Image for Detection")
        
        # File uploader with clear instructions
        st.markdown("""
        <div class="upload-box">
            <h3> Drop your image here or click to browse</h3>
            <p>Supported formats: JPG, JPEG, PNG</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image containing faces to detect masks",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            # Create two columns for before/after
            col1, col2 = st.columns(2)
            
            # Load and display original image
            image = Image.open(uploaded_file)
            
            with col1:
                st.markdown("###  Original Image")
                st.image(image, use_column_width=True)
            
            # Process image
            with st.spinner(" Detecting faces and masks..."):
                output_image, results = detect_and_predict(
                    image, model, class_labels, face_cascade
                )
            
            with col2:
                st.markdown("###  Detection Results")
                st.image(output_image, use_column_width=True)
            
            # Display results
            st.markdown("---")
            st.subheader(" Detection Summary")
            
            if len(results) == 0:
                st.warning(" No faces detected in the image. Try uploading a different image with clear faces.")
            else:
                # Statistics
                with_mask = sum(1 for r in results if r['label'] == 'with_mask')
                without_mask = len(results) - with_mask
                
                # Display stats in columns
                stat_col1, stat_col2, stat_col3 = st.columns(3)
                
                with stat_col1:
                    st.metric(" Total Faces", len(results))
                
                with stat_col2:
                    st.metric(" With Mask", with_mask)
                
                with stat_col3:
                    st.metric(" Without Mask", without_mask)
                
                # Detailed results
                st.markdown("###  Detailed Results")
                for i, result in enumerate(results, 1):
                    status_emoji = "" if result['label'] == 'with_mask' else "❌"
                    st.write(f"{status_emoji} **Face {i}:** {result['label'].replace('_', ' ').title()} - Confidence: {result['confidence']:.2f}%")
        else:
            st.info(" Please upload an image to start detection")
    
    # Mode 2: Webcam Capture
    elif detection_mode == " Capture from Webcam":
        st.subheader(" Capture Image from Webcam")
        
        st.info(" Click the button below to capture an image from your webcam")
        
        # Camera input
        camera_photo = st.camera_input("Take a picture")
        
        if camera_photo is not None:
            # Load image from camera
            image = Image.open(camera_photo)
            
            st.markdown("###  Processing Captured Image...")
            
            # Process image
            with st.spinner("Detecting faces and masks..."):
                output_image, results = detect_and_predict(
                    image, model, class_labels, face_cascade
                )
            
            # Display result
            st.image(output_image, caption="Detection Result", use_column_width=True)
            
            # Display statistics
            st.markdown("---")
            st.subheader(" Detection Summary")
            
            if len(results) == 0:
                st.warning(" No faces detected. Please adjust camera position and try again.")
            else:
                with_mask = sum(1 for r in results if r['label'] == 'with_mask')
                without_mask = len(results) - with_mask
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(" Faces", len(results))
                with col2:
                    st.metric(" With Mask", with_mask)
                with col3:
                    st.metric(" Without Mask", without_mask)
                
                # Detailed results
                st.markdown("### Results")
                for i, result in enumerate(results, 1):
                    status_emoji = "" if result['label'] == 'with_mask' else "❌"
                    st.write(f"{status_emoji} **Face {i}:** {result['label'].replace('_', ' ').title()} ({result['confidence']:.1f}%)")
    
    # Mode 3: Batch Processing
    elif detection_mode == " Batch Processing":
        st.subheader(" Batch Image Processing")
        
        st.markdown("""
        <div class="upload-box">
            <h3> Upload Multiple Images</h3>
            <p>Select multiple image files for batch processing</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Upload multiple images", 
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Upload multiple images for batch processing",
            label_visibility="collapsed"
        )
        
        if uploaded_files and len(uploaded_files) > 0:
            st.success(f" {len(uploaded_files)} images uploaded successfully!")
            
            if st.button(" Start Batch Processing", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                all_results = []
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    image = Image.open(uploaded_file)
                    output_image, results = detect_and_predict(
                        image, model, class_labels, face_cascade
                    )
                    
                    all_results.append({
                        'filename': uploaded_file.name,
                        'faces': len(results),
                        'with_mask': sum(1 for r in results if r['label'] == 'with_mask'),
                        'without_mask': sum(1 for r in results if r['label'] == 'without_mask')
                    })
                    
                    # Update progress
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                progress_bar.empty()
                status_text.empty()
                
                st.success(" Batch processing completed!")
                
                # Display summary
                st.markdown("---")
                st.subheader(" Batch Processing Summary")
                
                total_faces = sum(r['faces'] for r in all_results)
                total_with_mask = sum(r['with_mask'] for r in all_results)
                total_without_mask = sum(r['without_mask'] for r in all_results)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(" Images", len(uploaded_files))
                with col2:
                    st.metric(" Faces", total_faces)
                with col3:
                    st.metric(" With Mask", total_with_mask)
                with col4:
                    st.metric(" Without Mask", total_without_mask)
                
                # Detailed table
                st.markdown("###  Detailed Results")
                
                import pandas as pd
                df = pd.DataFrame(all_results)
                st.dataframe(df, use_container_width=True)
                
                # Download results
                csv = df.to_csv(index=False)
                st.download_button(
                    label=" Download Results (CSV)",
                    data=csv,
                    file_name="mask_detection_results.csv",
                    mime="text/csv"
                )
        else:
            st.info(" Please upload multiple images to start batch processing")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with  using Streamlit and TensorFlow | Face Mask Detection System</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()