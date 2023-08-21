def main():
    input_file = 'words.txt'
    output_file = 'words.txt'

    # Read words from input file
    with open(input_file, 'r') as file:
        words = file.readlines()

    # Remove duplicates by converting to set
    unique_words = set(words)

    # Write unique words to output file
    with open(output_file, 'w') as file:
        for word in unique_words:
            file.write(word)

    print(f"Unique words written to {output_file}")

if __name__ == "__main__":
    main()
