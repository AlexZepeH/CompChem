def extract_text(input_file, output_file, search_string, n):
    try:
        with open(input_file, 'r') as f:
            text = f.read()

        index = text.find(search_string)
        if index != -1:
            extracted_text = text[index + len(search_string):index + len(search_string) + n]

            with open(output_file, 'w') as output_f:
                output_f.write(extracted_text)

            print(f"Extracted text: {extracted_text}")
        else:
            print("Search string not found in the input file.")

    except FileNotFoundError:
        print("Input file not found.")

# Example usage
input_file = "input.txt"  # Replace with your input file's path
output_file = "output.txt"  # Replace with your output file's path
search_string = "example"  # Replace with the string you're searching for
n = 50  # Replace with the number of characters you want to extract

extract_text(input_file, output_file, search_string, n)