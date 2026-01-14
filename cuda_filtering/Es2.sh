#!/bin/bash
if [ "$#" -ne 2 ]; then
    echo "Error: Incorrect number of arguments."
    echo "Usage: $0 <input_directory> <output_directory>"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"

#mixto Compilation
export PATH=/usr/local/cuda-11.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH
echo "----------  Starting Compiling  ------------------"

nvcc Es2.cu -o Es2 \
  -I/usr/include/opencv4 \
  -L/usr/lib/aarch64-linux-gnu \
  -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc \
  -lstdc++ -lcudart

# Run for each image in noise directory
mkdir -p ../logData/nsysProfile
touch ../logData/logFile
counter=0
while [ $counter -lt 1 ]; do
    echo "Processing images in iteration $counter"
    counterimg=0
    for img in "$INPUT_DIR"/*.jpg; do
        if [ -f "$img" ]; then
            if (( counterimg % 10 == 0 )); then
                echo "Processing image $counterimg"
            fi
            img_directory=$(dirname "$img")
            img_filename=$(basename "$img")
            nsys profile \
            --trace=cuda \
            --cuda-memory-usage=true\
            -o ../logData/nsysProfile/${img_filename}.nsys-rep \
            --force-overwrite true \
            ./Es2 "$INPUT_DIR"/"$img_filename" "$OUTPUT_DIR"/BLUR"$img_filename" > /dev/null 2>&1
            echo "Image processed: $img_filename" >> ../logData/logFile
            mkdir -p ../logData/nsysProfile/reportCSV${counter}
            nsys stats -f csv -o ../logData/nsysProfile/reportCSV${counter}/${img_filename} -r gpumemsizesum  ../logData/nsysProfile/${img_filename}.nsys-rep > /dev/null 2>&1
            rm ../logData/nsysProfile/${img_filename}.nsys-rep
            rm ../logData/nsysProfile/${img_filename}.sqlite
            counterimg=$((counterimg+1))
        else
            echo "No images found in $INPUT_DIR."
        fi
    done
    counter=$((counter+1))
done
rm ./Es2
