import numpy as np
from PIL import Image
import cv2
from typing import List, Dict

def analyze_color_palette(image: Image.Image, num_colors: int = 3) -> List[Dict[str, str]]:
    """Extracts dominant color palette from an outfit image using K-Means clustering."""
    img_np = np.array(image.resize((150, 150)))
    pixels = img_np.reshape(-1, 3)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels.astype(np.float32), num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    _, counts = np.unique(labels, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]
    
    palette = []
    total_pixels = len(pixels)
    for idx in sorted_indices:
        color = centers[idx].astype(int)
        percentage = round(float(counts[idx] / total_pixels) * 100, 1)
        hex_color = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
        palette.append({
            "hex": hex_color,
            "rgb": f"rgb({color[0]}, {color[1]}, {color[2]})",
            "percentage": percentage
        })
    return palette