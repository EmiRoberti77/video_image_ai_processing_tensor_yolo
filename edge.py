import cv2
import numpy as np
from matplotlib import pyplot as plt

def extract_facial_markings(image_path, output_path):
    
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(image.shape[:2], dtype="uint8")

    
    for contour in contours:
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    
    extracted_markings = cv2.bitwise_and(image, image, mask=mask)
    cv2.imwrite(output_path, extracted_markings)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Extracted Markings')
    plt.imshow(cv2.cvtColor(extracted_markings, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()

# Example usage
image_path = 'images/horse.jpg'  # Path to your image
output_path = 'extracted_markings.png'  # Path to save the extracted markings
extract_facial_markings(image_path, output_path)
