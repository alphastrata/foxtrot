#!/bin/bash

# Script to generate thumbnails from all camera views (both Foxtrot and OCCT engines)
# Usage: ./all_views.sh <input_step_file> <output_directory>
RUST_LOG=off,step_thumbnailer=trace

set -e  # Exit on any error

if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_step_file> <output_directory>"
    echo "Example: $0 model.step ./output/"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_DIR="$2"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Array of all camera views
VIEWS=("isometric" "top" "bottom" "left" "right" "front" "back")

echo "Generating thumbnails for all views from: $INPUT_FILE"

# Generate thumbnails with Foxtrot engine
echo "=== Generating Foxtrot thumbnails ==="
for view in "${VIEWS[@]}"; do
    OUTPUT_FILE="$OUTPUT_DIR/$(basename "$INPUT_FILE" .step)_${view}.png"
    echo "Generating $view view with Foxtrot engine..."
    #NOTE: the decimation ratio here is tuned to be optically close what OCCT is using to make things 'fair'
    cargo run --release --bin step_thumbnailer -- -i "$INPUT_FILE" -o "$OUTPUT_FILE" -s 512 --view "$view" --engine foxtrot --decimation-ratio=0.18 
done

# Generate thumbnails with OCCT engine
echo "=== Generating OCCT thumbnails ==="
for view in "${VIEWS[@]}"; do
    OUTPUT_FILE="$OUTPUT_DIR/$(basename "$INPUT_FILE" .step)_${view}_occt.png"
    echo "Generating $view view with OCCT engine..."
    cargo run --release --bin step_thumbnailer --features occt -- -i "$INPUT_FILE" -o "$OUTPUT_FILE" -s 512 --view "$view" --engine occt
done

echo "All thumbnail views generated in: $OUTPUT_DIR"
echo "Files with '_occt.png' suffix were generated with the OCCT engine"

# Function to check for identical files
check_identical_files() {
    local output_dir="$1"
    local files=("$output_dir"/*.png)
    
    if [ ! -d "$output_dir" ] || [ -z "$(ls -A "$output_dir"/*.png 2>/dev/null)" ]; then
        echo "No PNG files found in $output_dir"
        return 0
    fi
    
    local all_files=("$output_dir"/*.png)
    local num_files=${#all_files[@]}
    
    echo "Checking for identical files among $num_files PNG files..."
    
    # Track identical pairs
    local identical_found=0
    
    for ((i = 0; i < num_files; i++)); do
        for ((j = i + 1; j < num_files; j++)); do
            if [ -f "${all_files[i]}" ] && [ -f "${all_files[j]}" ]; then
                if cmp -s "${all_files[i]}" "${all_files[j]}"; then
                    # Extract view names from filenames for clearer reporting
                    local file1_basename=$(basename "${all_files[i]}")
                    local file2_basename=$(basename "${all_files[j]}")
                    
                    echo "ERROR: Found identical files: ${all_files[i]} and ${all_files[j]}"
                    echo "       File 1: $file1_basename"
                    echo "       File 2: $file2_basename"
                    
                    # Extract view information from filenames
                    local view1=$(echo "$file1_basename" | sed -E 's/.*_([a-z]+)(_[a-z]+)?\.png/\1/')
                    local view2=$(echo "$file2_basename" | sed -E 's/.*_([a-z]+)(_[a-z]+)?\.png/\1/')
                    
                    echo "       Views: '$view1' vs '$view2'"
                    echo "This indicates that different views may not be generating correctly."
                    
                    identical_found=1
                fi
            fi
        done
    done
    
    if [ $identical_found -eq 1 ]; then
        echo "Exiting due to identical thumbnails found."
        exit 1
    else
        echo "No identical files found. All thumbnails appear to be unique."
    fi
}

# Check for identical files
check_identical_files "$OUTPUT_DIR"

# Function to check for potential isometric view issues
check_isometric_issues() {
    local output_dir="$1"
    echo "Checking for potential isometric view issues..."
    
    # Count isometric and non-isometric files separately
    local iso_count=0
    local non_iso_count=0
    local iso_total_size=0
    local non_iso_total_size=0
    
    # Loop through all PNG files in the directory
    for file in "$output_dir"/*.png; do
        if [[ -f "$file" ]]; then
            local file_size=$(stat -c%s "$file")
            if [[ "$file" == *isometric* ]]; then
                ((iso_count++))
                iso_total_size=$((iso_total_size + file_size))
                echo "Isometric file: $(basename "$file") - $file_size bytes"
            else
                ((non_iso_count++))
                non_iso_total_size=$((non_iso_total_size + file_size))
                echo "Non-isometric file: $(basename "$file") - $file_size bytes"
            fi
        fi
    done
    
    echo "Summary: $iso_count isometric files, $non_iso_count non-isometric files"
    
    if [ $iso_count -gt 0 ]; then
        local avg_iso_size=$((iso_total_size / iso_count))
        echo "Average isometric file size: $avg_iso_size bytes"
    fi
    
    if [ $non_iso_count -gt 0 ]; then
        local avg_non_iso_size=$((non_iso_total_size / non_iso_count))
        echo "Average non-isometric file size: $avg_non_iso_size bytes"
    fi
    
    # Check if isometric files are suspiciously small compared to others
    if [ $iso_count -gt 0 ] && [ $non_iso_count -gt 0 ]; then
        local avg_iso_size=$((iso_total_size / iso_count))
        local avg_non_iso_size=$((non_iso_total_size / non_iso_count))
        
        if [ $avg_iso_size -lt $((avg_non_iso_size * 70 / 100)) ]; then
            echo "WARNING: Isometric view files are significantly smaller on average than other views."
            echo "This may indicate that the isometric views are not fully rendering the part."
            echo "Consider checking the camera position or field of view for isometric views."
        else
            echo "Isometric view files appear to be of normal size compared to other views."
        fi
    fi
}

# Check for potential isometric view issues
check_isometric_issues "$OUTPUT_DIR"