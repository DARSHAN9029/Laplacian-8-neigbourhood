import numpy as np
from PIL import Image

def laplacian_8_neighborhood(image_path, output_path):
    image = Image.open(image_path)      #image loading

    max_size = 512  # size of image
    if max(image.size) > max_size:
        image = image.resize((max_size, max_size))

   
    image_array = np.array(image, dtype=np.float32) #image conversion to an array using numpy

    laplacian_kernel = np.array([[1, 1, 1],                                 # defining Laplacain 8 neighborhood matrix
                                 [1, -8, 1],
                                 [1, 1, 1]], dtype=np.float32)

    rows, cols = image_array.shape                              # Get image dimensions

    # Creating output array
    output_image = np.zeros_like(image_array)


    output_image[1:-1, 1:-1] = (
        image_array[:-2, :-2] * laplacian_kernel[0, 0] +  # Top-left
        image_array[:-2, 1:-1] * laplacian_kernel[0, 1] +  # Top
        image_array[:-2, 2:] * laplacian_kernel[0, 2] +  # Top-right
        image_array[1:-1, :-2] * laplacian_kernel[1, 0] +  # Left
        image_array[1:-1, 1:-1] * laplacian_kernel[1, 1] +  # Center
        image_array[1:-1, 2:] * laplacian_kernel[1, 2] +  # Right
        image_array[2:, :-2] * laplacian_kernel[2, 0] +  # Bottom-left
        image_array[2:, 1:-1] * laplacian_kernel[2, 1] +  # Bottom
        image_array[2:, 2:] * laplacian_kernel[2, 2]    # Bottom-right
    )

    # Normalize and clip values to prevent overflow
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)

    # Convert back to image and save
    output_image_pil = Image.fromarray(output_image)
    output_image_pil.save(output_path)

# Example usage
input_image_path = "input.jpg"  # Replace with actual path
output_image_path = "laplacian_output.jpg"

laplacian_8_neighborhood(input_image_path, output_image_path)

print(f"Laplacian edge-detected image saved at: {output_image_path}")