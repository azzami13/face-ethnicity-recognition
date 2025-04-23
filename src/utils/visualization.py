import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import io
from PIL import Image

from src.config import ETHNICITY_MAPPING_REVERSE

def visualize_comparison(img1, img2, box1, box2, similarity, threshold=0.6):
    """
    Visualize face comparison result
    
    Args:
        img1: First image
        img2: Second image
        box1: Bounding box for first face [x, y, w, h]
        box2: Bounding box for second face [x, y, w, h]
        similarity: Similarity score
        threshold: Threshold for determining if same person
        
    Returns:
        numpy.ndarray: Visualization image
    """
    # Create a copy of the images
    img1_copy = img1.copy()
    img2_copy = img2.copy()
    
    # Draw bounding boxes
    color = (0, 255, 0) if similarity >= threshold else (0, 0, 255)
    
    # First image
    x1, y1, w1, h1 = box1
    cv2.rectangle(img1_copy, (x1, y1), (x1 + w1, y1 + h1), color, 2)
    
    # Second image
    x2, y2, w2, h2 = box2
    cv2.rectangle(img2_copy, (x2, y2), (x2 + w2, y2 + h2), color, 2)
    
    # Resize images to same height
    height = min(img1_copy.shape[0], img2_copy.shape[0])
    width1 = int(img1_copy.shape[1] * (height / img1_copy.shape[0]))
    width2 = int(img2_copy.shape[1] * (height / img2_copy.shape[0]))
    
    img1_resized = cv2.resize(img1_copy, (width1, height))
    img2_resized = cv2.resize(img2_copy, (width2, height))
    
    # Create result image with both images side by side
    result_width = width1 + width2 + 100  # Add padding between images
    result_height = height + 50  # Add space for text
    result = np.ones((result_height, result_width, 3), dtype=np.uint8) * 255
    
    # Place first image
    result[25:25+height, 25:25+width1] = img1_resized
    
    # Place second image
    result[25:25+height, 25+width1+50:25+width1+50+width2] = img2_resized
    
    # Draw similarity score
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Similarity: {similarity:.2f}"
    text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
    text_x = (result_width - text_size[0]) // 2
    text_y = result_height - 15
    
    cv2.putText(result, text, (text_x, text_y), font, 0.7, 
                (0, 0, 0), 2, cv2.LINE_AA)
    
    # Add match/no match text
    match_text = "MATCH" if similarity >= threshold else "NO MATCH"
    match_color = (0, 128, 0) if similarity >= threshold else (0, 0, 128)
    
    text_size = cv2.getTextSize(match_text, font, 1.0, 2)[0]
    text_x = (result_width - text_size[0]) // 2
    text_y = 20
    
    cv2.putText(result, match_text, (text_x, text_y), font, 1.0, 
                match_color, 2, cv2.LINE_AA)
    
    return result

def visualize_ethnicity(image, box, ethnicity, probabilities):
    """
    Visualize ethnicity detection result
    
    Args:
        image: Input image
        box: Bounding box for face [x, y, w, h]
        ethnicity: Predicted ethnicity
        probabilities: Class probabilities
        
    Returns:
        numpy.ndarray: Visualization image
    """
    # Create a copy of the image
    img_copy = image.copy()
    
    # Draw bounding box
    x, y, w, h = box
    cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Draw ethnicity label
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Ethnicity: {ethnicity}"
    cv2.putText(img_copy, text, (x, y - 10), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Create bar chart for probabilities
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Get top 5 ethnicities
    indices = np.argsort(probabilities)[::-1][:5]
    top_probs = probabilities[indices]
    top_labels = [ETHNICITY_MAPPING_REVERSE.get(i, f"Unknown {i}") for i in indices]
    
    ax.bar(top_labels, top_probs, color='skyblue')
    ax.set_ylabel('Probability')
    ax.set_title('Top 5 Ethnicity Predictions')
    ax.set_ylim(0, 1)
    
    # Save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    
    # Convert buffer to numpy array
    plot_img = np.array(Image.open(buf))
    
    # Resize plot if needed
    plot_height = img_copy.shape[0] // 2
    plot_width = int(plot_img.shape[1] * (plot_height / plot_img.shape[0]))
    plot_img_resized = cv2.resize(plot_img, (plot_width, plot_height))
    
    # Create result image
    result_height = img_copy.shape[0] + plot_height
    result_width = max(img_copy.shape[1], plot_width)
    result = np.ones((result_height, result_width, 3), dtype=np.uint8) * 255
    
    # Place image
    result[:img_copy.shape[0], :img_copy.shape[1]] = img_copy
    
    # Place plot
    plot_y = img_copy.shape[0]
    plot_x = (result_width - plot_width) // 2
    result[plot_y:plot_y+plot_height, plot_x:plot_x+plot_width] = plot_img_resized[:, :, :3]
    
    return result