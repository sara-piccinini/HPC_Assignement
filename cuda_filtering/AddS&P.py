import random
import cv2
import sys
import os

def add_noise(img, amount):
    # Getting the dimensions of the image
    row, col, _ = img.shape  # Use this for color images
    
    # Calculate the total number of pixels to change
    total_pixels = int((amount / 100) * (row * col))
    
    # Salt (white pixels)
    num_salt = total_pixels // 2
    for _ in range(num_salt):
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)
        img[y_coord][x_coord] = [255, 255, 255]  # Set to white
        
    # Pepper (black pixels)
    num_pepper = total_pixels - num_salt
    for _ in range(num_pepper):
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)
        img[y_coord][x_coord] = [0, 0, 0]  # Set to black
        
    return img
        
def main():
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <image_file> <amount_percent>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    try:
        amount = int(sys.argv[2])
    except ValueError:
        print("Error: Amount must be a number.")
        sys.exit(1)

    # Load the image in COLOR
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: Could not read image file: {image_path}")
        sys.exit(1)
        
    noisy_img = add_noise(img, amount)
    
    # Create a new filename for the noisy image
    base_name = os.path.basename(image_path)
    output_filename = f'./input/noise{amount}/noisy{amount}_' + base_name
    cv2.imwrite(output_filename, noisy_img)
    print(f"Saved noisy image as {output_filename}")

if __name__ == "__main__":
    main()