#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 file1 file2 line_number"
    exit 1
fi

file1="$1"
file2="$2"
line_number="$3"

# Check if files exist
if [ ! -e "$file1" ]; then
    echo "Error: $file1 does not exist."
    exit 1
fi

if [ ! -e "$file2" ]; then
    echo "Error: $file2 does not exist."
    exit 1
fi

# Create a temporary file with the content of file1.txt
temp_file=$(mktemp)

# Use head to get the lines up to the insertion point
head -n "$line_number" "$file2" > "$temp_file"

# Append the content of file1.txt to the temporary file
cat "$file1" >> "$temp_file"

# Use tail to get the remaining lines
tail -n +"$line_number" "$file2" >> "$temp_file"

# Output the result to a new file
output_file="output.txt"
mv "$temp_file" "$output_file"

echo "Contents of $file1 sandwiched by $file2 at line $line_number saved to $output_file."
