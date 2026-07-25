"""
Validate coating mask presence in dataset samples.

This script iterates over sample_* directories in the coating_dataset directory,
checks the coating_mask.png file in each sample, and identifies samples where
the coating mask has insufficient coverage (<= 5% non-black pixels).
"""

import os
import glob
from PIL import Image
import numpy as np
import argparse


def analyze_coating_mask(mask_path):
    """
    Analyze a coating mask image to determine the percentage of non-black pixels.
    
    Args:
        mask_path (str): Path to the coating mask PNG file
        
    Returns:
        float: Percentage of non-black pixels (0-100)
    """
    try:
        img = Image.open(mask_path)
        img_array = np.array(img)
        
        # Handle different image modes
        if len(img_array.shape) == 3:  # RGB or RGBA
            # Convert to grayscale by taking the maximum across color channels
            # This ensures any non-black pixel in any channel is detected
            grayscale = np.max(img_array[:, :, :3], axis=2)
        else:  # Already grayscale
            grayscale = img_array
        
        # Count non-black pixels
        black_pixel_threshold = 10
        non_black_pixels = np.sum(grayscale > black_pixel_threshold)
        total_pixels = grayscale.size
        
        # Calculate percentage
        percentage = (non_black_pixels / total_pixels) * 100.0
        
        return percentage

    except Exception as e:
        print(f"Error processing {mask_path}: {e}")
        return None


def validate_coating_masks(dataset_root, threshold=5.0, num_coatings=None, check_jsons=False, include_incomplete=False):
    """
    Validate coating masks in all sample directories.
    
    Args:
        dataset_root (str): Path to the dataset root directory
        threshold (float): Threshold percentage for mask coverage (default: 5.0%)
        num_coatings (int, optional): Number of coating files to validate (coating_0.png to coating_n-1.png)
        
    Returns:
        tuple: (bad_samples, total_samples)
    """
    dataset_path = os.path.abspath(dataset_root)
    
    if not os.path.exists(dataset_path):
        print(f"Dataset directory not found: {dataset_path}")
        return [], 0
    
    # Find all sample directories
    sample_pattern = os.path.join(dataset_path, "sample_*")
    sample_dirs = sorted(glob.glob(sample_pattern))
    
    if not sample_dirs:
        print(f"No sample directories found in: {dataset_path}")
        return [], 0
    
    print(f"Found {len(sample_dirs)} sample directories")
    print(f"Analyzing coating mask coverage (threshold: {threshold}%)")
    print("-" * 60)
    
    bad_samples = []
    processed_samples = 0
    
    for sample_dir in sample_dirs:
        sample_name = os.path.basename(sample_dir)
        try:
            sample_number = int(sample_name.split('_')[1])
        except (IndexError, ValueError):
            print(f"Warning: Could not extract sample number from {sample_name}")
            continue
        
        text_data_path = os.path.join(sample_dir, "text_data.json")

        if not os.path.exists(text_data_path) and check_jsons:
            print(f"Sample {sample_number:6d}: Missing text_data.json")

            if include_incomplete:
                bad_samples.append(sample_number)
                processed_samples += 1

            continue
        
        # Check for coating files if num_coatings is specified
        if num_coatings is not None:
            missing_coatings = []
            for i in range(num_coatings):
                coating_path = os.path.join(sample_dir, f"coating_{i}.png")
                if not os.path.exists(coating_path):
                    missing_coatings.append(i)
            
            if missing_coatings:
                print(f"Sample {sample_number:6d}: Missing coating files: {missing_coatings}")

                if include_incomplete:
                    bad_samples.append(sample_number)
                    processed_samples += 1

                continue
        
        mask_path = os.path.join(sample_dir, "coating_mask.png")
        
        if not os.path.exists(mask_path):
            print(f"Sample {sample_number:6d}: Missing coating_mask.png")

            if include_incomplete:
                bad_samples.append(sample_number)
                processed_samples += 1

            continue
        
        coverage_percentage = analyze_coating_mask(mask_path)
        
        if coverage_percentage is None:
            print(f"Sample {sample_number:6d}: Error analyzing mask")

            if include_incomplete:
                bad_samples.append(sample_number)
                processed_samples += 1
        elif coverage_percentage <= threshold:
            print(f"Sample {sample_number:6d}: Low coverage ({coverage_percentage:.2f}%)")
            bad_samples.append(sample_number)
        
        processed_samples += 1
        
        # Progress indicator
        if processed_samples % 100 == 0:
            print(f"Processed {processed_samples}/{len(sample_dirs)} samples...")
    
    return bad_samples, processed_samples


def main():
    parser = argparse.ArgumentParser(description="Validate coating mask presence in dataset samples")
    parser.add_argument("dataset_root", help="Path to the dataset root directory")
    parser.add_argument("--num-coatings", type=int, help="Number of coating files to validate (coating_0.png to coating_n-1.png)")
    parser.add_argument("--check-jsons", action="store_true", default=False, help="Check if coating mask files are present in the dataset root directory")
    parser.add_argument("--include-incomplete", action="store_true", default=False,
                        help="Include incomplete samples for the validation output")
    
    args = parser.parse_args()
    dataset_root = args.dataset_root
    
    print("=== Coating Mask Validation ===")
    bad_samples, total_samples = validate_coating_masks(dataset_root, threshold=1.0, num_coatings=args.num_coatings,
                                                        check_jsons=args.check_jsons,
                                                        include_incomplete=args.include_incomplete)

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total samples processed: {total_samples}")
    print(f"Bad coating samples: {len(bad_samples)}")
    
    if total_samples > 0:
        bad_percentage = (len(bad_samples) / total_samples) * 100
        print(f"Bad sample percentage: {bad_percentage:.2f}%")
    
    if bad_samples:
        print(f"\nBad sample numbers:")
        for i in range(0, len(bad_samples), 10):
            row = bad_samples[i:i+10]
            print("  " + ", ".join(f"{num:6d}" for num in row))
    else:
        print("\nAll samples have adequate coating mask coverage!")


if __name__ == "__main__":
    main()