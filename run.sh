

#!/bin/bash

#######################################
# Main Pipeline: Image Generation & Fusion
#######################################
##Environment Requirements:

# Install ImageMagick:
# Ubuntu/Debian: sudo apt-get install imagemagick
# CentOS/RHEL: sudo yum install ImageMagick
# macOS: brew install imagemagick

# Verify installation: convert --version
#######################################

# Phase 1: Generate results1
echo "Generating results1..."
CUDA_VISIBLE_DEVICES=0 python3 test_demo.py \
--data_dir ./NTIRE2025_Challenge/input \
--model_id 2 \
--save_dir ./NTIRE2025_Challenge/results1

# Phase 2: Generate results2 (for model ensemble/testing)
echo "Generating results2..."
CUDA_VISIBLE_DEVICES=0 python3 test_demo.py \
--data_dir ./NTIRE2025_Challenge/input \
--model_id 2 \
--save_dir ./NTIRE2025_Challenge/results2

#######################################
# Phase 3: Image Fusion Module
#######################################

# Create output directory
mkdir -p ./NTIRE2025_Challenge/results3
echo "Output directory ready: ./NTIRE2025_Challenge/results3"

# Batch processing loop
echo "Starting image fusion..."
count=0
for file in ./NTIRE2025_Challenge/results1/*; do
    # Extract filename (handles special characters)
    filename=$(basename "$file")
    
    # File path construction
    result1_file="./NTIRE2025_Challenge/results1/$filename"
    result2_file="./NTIRE2025_Challenge/results2/$filename"
    output_file="./NTIRE2025_Challenge/results3/$filename"

    #######################################
    # Core Fusion Logic (ImageMagick implementation)
    # Parameters:
    # -colorspace RGB       : Unified color space
    # -define compose:args=50,50 : 50-50 blend weights
    # -compose blend        : Blending algorithm
    # -composite            : Execute composition
    # -colorspace sRGB      : Convert to standard color space
    #######################################
    convert "$result1_file" "$result2_file" \
        -colorspace RGB \
        -define compose:args=50,50 \
        -compose blend \
        -composite \
        -colorspace sRGB \
        "$output_file"

    # Progress indicator
    count=$((count+1))
    if (( $count % 10 == 0 )); then
        echo "Processed $count images..."
    fi
done

#######################################
# Post-processing
#######################################
echo "Fusion completed! Total processed: $count images"
echo "Final results saved to: ./NTIRE2025_Challenge/results3"

# Optional: Set file permissions
# chmod 755 ./NTIRE2025_Challenge/results3/*

