import os
import re


def extract_words_from_filename(filename):
    # Remove file extension and split by hyphen
    filename = os.path.splitext(filename)[0]
    words = re.findall(r'\w+', filename)
    return words

def main():
    folder_path = '/remote-home/pjli/codefield/openai/leetcode-problemset/leetcode/problem'
    output_file = 'words.txt'

    # Get list of filenames in the folder
    filenames = os.listdir(folder_path)

    # Extract words from each filename
    all_words = []
    for filename in filenames:
        words = extract_words_from_filename(filename)
        all_words.extend(words)

    # Remove duplicates and sort the words
    unique_words = sorted(set(all_words))

    # Append unique words to output file without overwriting
    with open(output_file, 'a') as file:
        for word in unique_words:
            file.write(word + '\n')

    print(f"Extracted words appended to {output_file}")

if __name__ == "__main__":
    main()
