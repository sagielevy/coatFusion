import numpy as np
from PIL import Image
import random

def crop_images_to_bounding_box_by_mask(logger, mask_image, desired_resolution, idx, jitter=False):
    """
    Return bounds to crop a mask and other images to a fixed-size square that captures the most white pixels.

    Args:
        logger: Logger instance for warning messages
        mask_image: PIL Image of the mask
        desired_resolution: Target resolution (square output, fixed size)
        idx: Index for logging purposes
        jitter: True if should jitter

    Returns:
        Tuple of (crop_x_min, crop_y_min, crop_x_max, crop_y_max)
    """
    # Convert mask to grayscale
    mask = mask_image.convert('L')
    mask_array = np.array(mask)
    img_height, img_width = mask_array.shape

    if desired_resolution > min(img_width, img_height):
        raise ValueError(f"Desired resolution {desired_resolution} exceeds image dimensions {img_width}x{img_height}")

    # Apply threshold: <10 = black, else white
    binary_mask = (mask_array >= 10).astype(np.int32)

    if not np.any(binary_mask):
        raise ValueError(f"No white regions found in mask for idx {idx}")

    # Calculate valid crop ranges
    max_crop_x = img_width - desired_resolution
    max_crop_y = img_height - desired_resolution

    # Generate random 2D jitter for cropping (10% of original image size or part of the resolution)
    original_width, original_height = mask_image.size
    jitter_range_x = min(int(original_width * 0.1), desired_resolution // 2)
    jitter_range_y = min(int(original_height * 0.1), desired_resolution // 2)
    jitter_x = random.randint(-jitter_range_x, jitter_range_x)
    jitter_y = random.randint(-jitter_range_y, jitter_range_y)

    if not jitter:
        jitter_x = jitter_y = jitter_range_x = jitter_range_y = 0

    stride = max(1, desired_resolution // 2)

    max_white_pixels = 0
    best_crop_x = 0
    best_crop_y = 0

    for crop_y in range(0, max_crop_y + 1, stride):
        for crop_x in range(0, max_crop_x + 1, stride):
            # Count white pixels in this crop window
            crop_region = binary_mask[crop_y:crop_y + desired_resolution,
                          crop_x:crop_x + desired_resolution]
            white_pixel_count = np.sum(crop_region)

            if white_pixel_count > max_white_pixels:
                max_white_pixels = white_pixel_count
                best_crop_x = crop_x
                best_crop_y = crop_y

    jitterd_best_crop_x = max(0, min(best_crop_x + jitter_x, max_crop_x))
    jitterd_best_crop_y = max(0, min(best_crop_y + jitter_y, max_crop_y))
    
    crop_bounds = (jitterd_best_crop_x, jitterd_best_crop_y,
                   jitterd_best_crop_x + desired_resolution,
                   jitterd_best_crop_y + desired_resolution)

    if not np.any(Image.fromarray(binary_mask).crop(crop_bounds)):
        if logger:
            logger.warning(f"No white regions found in mask after editing index {idx}, bounds: {crop_bounds}. falling back to no jitter")

        best_crop_x = max(0, min(best_crop_x, max_crop_x))
        best_crop_y = max(0, min(best_crop_y, max_crop_y))

        crop_bounds = (best_crop_x, best_crop_y,
                       best_crop_x + desired_resolution,
                       best_crop_y + desired_resolution)

        if not np.any(Image.fromarray(binary_mask).crop(crop_bounds)):
            if logger:
                logger.warning(f"No white regions found in fallback bounds for index {idx}")
        return crop_bounds

    return crop_bounds


# test
if __name__ == "__main__":
    mask_path = "coating_dataset/sample_1003/coating_mask.png"
    coating_path = "coating_dataset/sample_1003/coating_2.png"
    output_path = "coating_dataset/test/"
    mask_img = Image.open(mask_path)
    coating_img = Image.open(coating_path)
    cropped_images = crop_images_to_bounding_box_by_mask(None, mask_img, 512, 0)

    for index, img in enumerate(cropped_images):
        img.save(output_path + f"{index}.png")