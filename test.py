def extract_text(input_file, output_file, search_string, n):
    try:
        with open(input_file, 'r') as f:
            text = f.read()

        index = text.find(search_string)
        if index != -1:
            extracted_text = text[index + len(search_string) + 23 :index + 24 + len(search_string) + n]

            print(f"Extracted text: {extracted_text}")
            return extracted_text
        else:
            print("Search string not found in the input file.")

    except FileNotFoundError:
        print("Input file not found.")

