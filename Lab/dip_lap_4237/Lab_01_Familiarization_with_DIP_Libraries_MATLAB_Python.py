import cv2
import matplotlib.pyplot as plt
from PIL import Image
from skimage import exposure

# Load grayscale image using OpenCV
img_cv = cv2.imread('sample_image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian Blur
img_blur = cv2.GaussianBlur(img_cv, (5, 5), 0)

# Histogram Equalization using skimage
img_eq = exposure.equalize_hist(img_cv)

# Display original, blurred, and equalized images
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1), plt.title("Original"), plt.imshow(img_cv, cmap='gray'), plt.axis('off')
plt.subplot(1, 3, 2), plt.title("Blurred"), plt.imshow(img_blur, cmap='gray'), plt.axis('off')
plt.subplot(1, 3, 3), plt.title("Equalized"), plt.imshow(img_eq, cmap='gray'), plt.axis('off')
plt.show()

# Load and display image using PIL
img_pil = Image.open('sample_image.jpg')
img_pil_gray = img_pil.convert('L')

# Display color and grayscale versions
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1), plt.title("Original (PIL)"), plt.imshow(img_pil), plt.axis('off')
plt.subplot(1, 2, 2), plt.title("Grayscale (PIL)"), plt.imshow(img_pil_gray, cmap='gray'), plt.axis('off')
plt.show()
