import os
import pytesseract
import pandas as pd
from pdf2image import convert_from_path
import cv2
import numpy as np 
from PIL import Image, ImageEnhance


# Function to convert PDF to images
def pdf_to_images(pdf_path, output_folder):
    pages = convert_from_path(pdf_path, 300)  # 300 DPI for better quality
    image_paths = []
    for i, page in enumerate(pages):
        image_path = os.path.join(output_folder, f"page_{i + 1}.png")
        page.save(image_path, "PNG")
        image_paths.append(image_path)
    return image_paths


# Function to apply thresholding to the image
def thresholding_img(img):
    _, thres1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thres1


# Function to apply noise reduction
def noise_reduction(img):
    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    median_blur = cv2.medianBlur(gray_img, 5)  # Apply median blur
    gaussian_blur = cv2.GaussianBlur(median_blur, (5, 5), 7, cv2.BORDER_DEFAULT)  # Apply Gaussian blur
    return gaussian_blur


# Preprocess images before sending to OCR
def preprocessing_image(image_path):
    img = cv2.imread(image_path)
    noise_reduced_image = noise_reduction(img)  # Applying noise reduction
    thresholded_img = thresholding_img(noise_reduced_image)

    # Enhance contrast using PIL
    enhanced_image = Image.fromarray(thresholded_img)
    enhancer = ImageEnhance.Contrast(enhanced_image)
    contrast_image_enhanced = enhancer.enhance(2.3)  # Experimental contrast adjustment
    return contrast_image_enhanced


# Extract text and bounding boxes using Tesseract OCR
def extract_text_and_bounding_boxes(image_paths, output_csv):
    data = []

    for image_path in image_paths:
        # Preprocess the image
        preprocessed_image = preprocessing_image(image_path)

        # Convert PIL image to OpenCV format for OCR
        preprocessed_cv_image = cv2.cvtColor(np.array(preprocessed_image), cv2.COLOR_RGB2BGR)

        # Use pytesseract to extract data
        ocr_data = pytesseract.image_to_data(preprocessed_cv_image, output_type=pytesseract.Output.DICT)

        # Loop through each word detected by Tesseract
        for i in range(len(ocr_data['text'])):
            if ocr_data['text'][i].strip():  # Only process non-empty text
                word_data = {
                    'image_path': image_path,
                    'page_number': image_paths.index(image_path) + 1,
                    'text': ocr_data['text'][i],
                    'confidence': ocr_data['conf'][i],
                    'x': ocr_data['left'][i],
                    'y': ocr_data['top'][i],
                    'width': ocr_data['width'][i],
                    'height': ocr_data['height'][i]
                }
                data.append(word_data)

    # Save results to a CSV file
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Extracted data saved to {output_csv}")


# Example Usage
if __name__ == "__main__":
    pdf_path = "D:\Learn_365_ALL\R000021017_6838771_unlocked.pdf"
    output_folder = "output_images"
    output_csv = "extracted_text.csv"

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Convert PDF to images
    image_paths = pdf_to_images(pdf_path, output_folder)

    # Extract text and bounding boxes
    extract_text_and_bounding_boxes(image_paths, output_csv)
