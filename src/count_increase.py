import json
import sys
import os
from pathlib import Path

def count_increase_occurrences(json_file_path):
    try:
        # Read the JSON file
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        
        # Count "Improve" occurrences in the list of turn pairs
        count = 0
        if isinstance(data, list):
            for turn_pair in data:
                if isinstance(turn_pair, dict) and 'user_clarity' in turn_pair:
                    user_clarity = turn_pair['user_clarity']
                    if isinstance(user_clarity, dict) and 'change' in user_clarity:
                        if user_clarity['change'] == "Improve":  # Note: In your JSON it's "Improve" not "Increase"
                            count += 1
        
        return count
    
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found.")
        return -1
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in the file '{json_file_path}'.")
        return -1
    except Exception as e:
        print(f"An unexpected error occurred with file '{json_file_path}': {str(e)}")
        return -1

def process_directory(directory_path):
    # Convert to Path object for better path handling
    dir_path = Path(directory_path)
    
    if not dir_path.is_dir():
        print(f"Error: '{directory_path}' is not a valid directory.")
        return
    
    # Get all JSON files in the directory
    json_files = list(dir_path.glob('**/*.json'))
    
    if not json_files:
        print(f"No JSON files found in '{directory_path}'")
        return
    
    total_count = 0
    valid_files = 0
    
    # Process each JSON file
    for json_file in json_files:
        count = count_increase_occurrences(json_file)
        if count >= 0:  # Only include valid results
            print(f"{json_file.name}: {count} occurrences")
            total_count += count
            valid_files += 1
    
    # Calculate and print average
    if valid_files > 0:
        average = total_count / valid_files
        print(f"\nSummary:")
        print(f"Total files processed: {valid_files}")
        print(f"Total 'Improve' occurrences: {total_count}")
        print(f"Average occurrences per file: {average:.2f}")
    else:
        print("\nNo valid JSON files were processed.")

if __name__ == "__main__":
    # Interactive loop: prompt user repeatedly until 'q' or 'quit' to exit
    while True:
        user_input = input("Please enter the directory path to process (or 'q' to quit): ").strip()
        if user_input.lower() in ('q', 'quit'):
            print("Exiting.")
            break
        if not user_input:
            print("No directory path provided, please try again or enter 'q' to quit.")
            continue
        process_directory(user_input) 