import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.face_detection.mtcnn_detector import MTCNNDetector
from src.face_similarity.facenet_embedder import FaceNetEmbedder
from src.ethnicity_detection.cnn_classifier import CNNEthnicityClassifier
from src.utils.visualization import visualize_comparison, visualize_ethnicity
from src.config import ETHNICITY_MAPPING_REVERSE, FACE_SIMILARITY_THRESHOLD

def main():
    st.title("Face Similarity & Ethnicity Detection")
    
    # Initialize modules
    @st.cache_resource
    def load_modules():
        face_detector = MTCNNDetector()
        face_embedder = FaceNetEmbedder()
        ethnicity_classifier = CNNEthnicityClassifier()
        return face_detector, face_embedder, ethnicity_classifier
    
    face_detector, face_embedder, ethnicity_classifier = load_modules()
    
    # Create sidebar
    st.sidebar.title("Options")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Face Similarity", "Ethnicity Detection"])
    
    if app_mode == "Face Similarity":
        st.header("Face Similarity Analysis")
        col1, col2 = st.columns(2)
        
        # Upload images
        with col1:
            uploaded_file1 = st.file_uploader("Choose first image", type=["jpg", "jpeg", "png"])
            if uploaded_file1 is not None:
                image1 = Image.open(uploaded_file1)
                st.image(image1, caption="First Image", use_column_width=True)
                
        with col2:
            uploaded_file2 = st.file_uploader("Choose second image", type=["jpg", "jpeg", "png"])
            if uploaded_file2 is not None:
                image2 = Image.open(uploaded_file2)
                st.image(image2, caption="Second Image", use_column_width=True)
                
        # Similarity threshold
        threshold = st.slider("Similarity Threshold", 0.0, 1.0, FACE_SIMILARITY_THRESHOLD, 0.01)
                
        # Process button
        if st.button("Compare Faces") and uploaded_file1 is not None and uploaded_file2 is not None:
            with st.spinner("Processing..."):
                # Convert PIL images to numpy arrays
                img1_np = np.array(image1)
                img2_np = np.array(image2)
                
                # Detect faces
                faces1 = face_detector.extract_faces(img1_np)
                faces2 = face_detector.extract_faces(img2_np)
                
                if not faces1 or not faces2:
                    st.error("No faces detected in one or both images")
                else:
                    # Get first face from each image
                    face1 = faces1[0]['face']
                    face2 = faces2[0]['face']
                    
                    # Calculate similarity
                    similarity = face_embedder.compare_faces(face1, face2)
                    
                    # Visualize results
                    result_img = visualize_comparison(img1_np, img2_np, faces1[0]['box'], 
                                                     faces2[0]['box'], similarity, threshold)
                    st.image(result_img, caption=f"Similarity Score: {similarity:.4f}", 
                            use_column_width=True)
                    
                    # Provide interpretation
                    if similarity > threshold:
                        st.success(f"These appear to be the same person (score: {similarity:.4f})")
                    else:
                        st.warning(f"These appear to be different people (score: {similarity:.4f})")
    
    else:  # Ethnicity Detection
        st.header("Ethnicity Detection")
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process button
            if st.button("Detect Ethnicity"):
                with st.spinner("Processing..."):
                    # Convert PIL image to numpy array
                    img_np = np.array(image)
                    
                    # Detect faces
                    faces = face_detector.extract_faces(img_np)
                    
                    if not faces:
                        st.error("No faces detected in the image")
                    else:
                        # Process each detected face
                        for i, face_data in enumerate(faces):
                            face = face_data['face']
                            
                            # Predict ethnicity
                            class_idx, probabilities = ethnicity_classifier.predict(face)
                            
                            # Convert to readable format
                            ethnicity = ETHNICITY_MAPPING_REVERSE.get(class_idx, f"Unknown {class_idx}")
                            
                            # Visualize results
                            result_img = visualize_ethnicity(img_np, face_data['box'], ethnicity, probabilities)
                            st.image(result_img, caption=f"Face #{i+1} - Predicted Ethnicity: {ethnicity}", 
                                    use_column_width=True)
                            
                            # Show probability distribution
                            st.subheader(f"Probability Distribution for Face #{i+1}")
                            
                            # Create dataframe for probabilities
                            probs_df = pd.DataFrame({
                                'Ethnicity': [ETHNICITY_MAPPING_REVERSE.get(j, f"Unknown {j}") for j in range(len(probabilities))],
                                'Probability': probabilities
                            })
                            
                            # Sort by probability
                            probs_df = probs_df.sort_values('Probability', ascending=False).reset_index(drop=True)
                            
                            # Display top 5
                            st.dataframe(probs_df.head(5))

if __name__ == "__main__":
    main()