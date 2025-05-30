import os
import json
import glob
import re
from pathlib import Path

def load_json_file(file_path):
    """Load and parse a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(file_path, data):
    """Save data as a JSON file with proper formatting."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def extract_task_and_number_from_filename(filename):
    """
    Extract the task part and sequence number from a filename.
    
    Returns a tuple: (task_name, sequence_number)
    """
    print(f"Processing filename: {filename}")
    
    # Extract the sequence number
    base_name = os.path.splitext(filename)[0]
    match = re.search(r'_(\d+)$', base_name)
    sequence_number = match.group(1) if match else "1"
    print(f"  Extracted sequence number: {sequence_number}")
    
    # Find the part before "_dialogue_" as the complete task name
    dialogue_idx = filename.find("_dialogue_")
    if dialogue_idx > 0:
        task_part = filename[:dialogue_idx]
        print(f"  Extracted task from filename: {task_part}")
        return task_part, sequence_number
    
    # If "_dialogue_" pattern not found, try other methods
    parts = filename.split('_')
    if len(parts) >= 3:
        # Assume the last two parts are turns and number
        possible_task = '_'.join(parts[:-2])
        print(f"  Alternative task parsing: {possible_task}")
        return possible_task, sequence_number
    
    # If all methods fail, return the filename (without extension)
    print(f"  Unable to parse, using base filename: {base_name}")
    return base_name, sequence_number

def find_profile_file(task_part, user_number):
    """
    Find the user profile file based on task name and user number.
    
    1. Direct match: <task>_user_<number>.json
    2. Smart matching: Find the most similar profile file based on task name
    """
    if not task_part:
        print("  Task name is empty, cannot find profile file")
        return None
    
    print(f"Looking for profile file for task '{task_part}' and user number '{user_number}'")
    
    # Build expected profile filename format
    profile_file_pattern = f"{task_part}_user_{user_number}.json"
    profile_path = Path("profiles/users/basic")
    
    # 1. Try direct match
    for file_path in profile_path.glob("*.json"):
        if file_path.name.lower() == profile_file_pattern.lower():
            print(f"  Found exact match: {file_path}")
            return file_path
    
    # 2. Try partial matching with the same user number
    specific_user_pattern = f"*_user_{user_number}.json"
    user_specific_profiles = list(profile_path.glob(specific_user_pattern))
    
    if user_specific_profiles:
        best_match = None
        best_score = 0
        
        print(f"  Available profile files for user {user_number}: {len(user_specific_profiles)}")
        
        for file_path in user_specific_profiles:
            # Extract task name from profile filename
            file_task = re.sub(r'_user_\d+\.json$', '', file_path.name)
            
            # Calculate similarity between task names
            # Simple method: check substring inclusion
            if task_part.lower() in file_task.lower() or file_task.lower() in task_part.lower():
                # Calculate match score (by common words)
                task_words = set(task_part.lower().split('_'))
                file_words = set(file_task.lower().split('_'))
                common_words = task_words.intersection(file_words)
                score = len(common_words)
                
                if score > best_score:
                    best_score = score
                    best_match = file_path
                    print(f"  Found better match: {file_path} (score: {score})")
        
        if best_match:
            print(f"  Best match: {best_match} (score: {best_score})")
            return best_match
    
    # 3. Fallback to matching any user profile if specific user not found
    print(f"  No matching profile found for user {user_number}, trying any user profile")
    all_profile_files = list(profile_path.glob("*_user_*.json"))
    
    best_match = None
    best_score = 0
    
    print(f"  Available profile files: {len(all_profile_files)}")
    
    for file_path in all_profile_files:
        # Extract task name from profile filename
        file_task = re.sub(r'_user_\d+\.json$', '', file_path.name)
        
        # Calculate similarity between task names
        if task_part.lower() in file_task.lower() or file_task.lower() in task_part.lower():
            task_words = set(task_part.lower().split('_'))
            file_words = set(file_task.lower().split('_'))
            common_words = task_words.intersection(file_words)
            score = len(common_words)
            
            if score > best_score:
                best_score = score
                best_match = file_path
                print(f"  Found better match: {file_path} (score: {score})")
    
    if best_match:
        print(f"  Best match: {best_match} (score: {best_score})")
        return best_match
    
    # If no match found, use the first profile file
    if all_profile_files:
        default_file = all_profile_files[0]
        print(f"  No match found, using default profile file: {default_file}")
        return default_file
    
    print("  No profile files found")
    return None

def convert_file(input_file, output_file=None, default_profile=None):
    """Convert a single file, adding user profile information."""
    try:
        # Load input file
        data = load_json_file(input_file)
        
        # Extract filename
        file_name = os.path.basename(input_file)
        
        # Extract task part and user number from filename
        task_part, user_number = extract_task_and_number_from_filename(file_name)
        
        # Find corresponding profile file
        profile_file = find_profile_file(task_part, user_number)
        
        # If no profile file found, use default profile
        if not profile_file and default_profile:
            print(f"Using default profile: {default_profile}")
            profile_file = default_profile
        
        if not profile_file:
            print(f"Could not find profile file for task '{task_part}' and user number '{user_number}'")
            return False
        
        # Load profile data
        profile_data = load_json_file(profile_file)
        
        # Add profile data to original data
        data['metadata']['user_profile'] = profile_data
        
        # Save output file
        if output_file is None:
            # If output path not specified, create default path
            output_dir = os.path.dirname(input_file).replace('output', 'output_without_addprofile')
            os.makedirs(output_dir, exist_ok=True)
            
            # Use the same filename
            output_file = os.path.join(output_dir, os.path.basename(input_file))
        
        save_json_file(output_file, data)
        print(f"Converted file saved to: {output_file}")
        return True
    except Exception as e:
        print(f"Error processing file {input_file}: {str(e)}")
        return False

def convert_all_files(input_dir, output_dir=None, default_profile=None):
    """Batch convert all JSON files in a directory."""
    # Find all JSON files in the directory
    input_files = glob.glob(os.path.join(input_dir, "**/*.json"), recursive=True)
    
    success_count = 0
    failure_count = 0
    already_has_profile_count = 0
    failure_files = []
    
    print(f"Found {len(input_files)} files to process")
    
    for input_file in input_files:
        print(f"\nProcessing: {input_file}")
        
        # Check if file already has user profile
        try:
            data = load_json_file(input_file)
            if 'metadata' in data and 'user_profile' in data['metadata']:
                print(f"  File already has user profile, skipping...")
                already_has_profile_count += 1
                continue
        except Exception as e:
            print(f"  Error reading file: {str(e)}")
            failure_count += 1
            failure_files.append(input_file)
            continue
        
        if output_dir:
            # Maintain the same directory structure
            rel_path = os.path.relpath(input_file, input_dir)
            output_file = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
        else:
            output_file = None
            
        if convert_file(input_file, output_file, default_profile):
            success_count += 1
        else:
            failure_count += 1
            failure_files.append(input_file)
    
    print(f"\nConversion completed:")
    print(f"- Files already with profiles: {already_has_profile_count}")
    print(f"- Successfully added profiles: {success_count}")
    print(f"- Failed conversions: {failure_count}")
    print(f"- Total files processed: {len(input_files)}")
    
    # Log failed files for debugging
    if failure_count > 0:
        with open('conversion_failures.log', 'w') as f:
            for file in failure_files:
                f.write(f"{file}\n")
        print(f"Failed files are logged in conversion_failures.log")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert dialogue files without user profiles to include user profiles")
    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("--output", help="Output file or directory (optional)")
    parser.add_argument("--all", action="store_true", help="Process all files in the input directory")
    parser.add_argument("--default-profile", help="Default profile file path to use when no matching profile is found")
    
    args = parser.parse_args()
    
    default_profile = args.default_profile
    
    if args.all:
        convert_all_files(args.input, args.output, default_profile)
    else:
        convert_file(args.input, args.output, default_profile) 