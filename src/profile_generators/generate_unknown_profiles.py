import os
import json
import random
import shutil
from pathlib import Path

def generate_unknown_profiles(source_dir, percentages=[40, 60, 80]):
    """
    Generate new user profiles with varying percentages of unknown information.
    
    Args:
        source_dir: Directory containing source user profiles
        percentages: List of percentages of information to mark as unknown
    """
    # Create output directories
    base_dir = Path('profiles/users')
    for percentage in percentages:
        unknown_dir = base_dir / f'unknown_{percentage}percent'
        os.makedirs(unknown_dir, exist_ok=True)
    
    # Process all JSON files in the source directory
    for filename in os.listdir(source_dir):
        if not filename.endswith('.json'):
            continue
            
        file_path = os.path.join(source_dir, filename)
        
        # Load the user profile
        with open(file_path, 'r', encoding='utf-8') as f:
            user_profile = json.load(f)
        
        # Generate profiles with different unknown percentages
        for percentage in percentages:
            # Create a new profile with unknown fields
            new_profile = create_unknown_profile(user_profile, percentage)
            
            # Create new filename
            task_name = filename.split('_user_')[0]
            user_num = filename.split('_user_')[1]
            new_filename = f"{task_name}_unknown{percentage}_user_{user_num}"
            
            # Save to the appropriate directory
            output_dir = base_dir / f'unknown_{percentage}percent'
            output_path = output_dir / new_filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(new_profile, f, indent=4)
                
            print(f"Generated {percentage}% unknown profile: {output_path}")

def create_unknown_profile(user_profile, percentage):
    """
    Create a new profile with a percentage of fields set to unknown.
    
    Args:
        user_profile: Original user profile
        percentage: Percentage of fields to mark as unknown
        
    Returns:
        Dict: New user profile with unknown fields
    """
    # Create a deep copy of the profile
    new_profile = json.loads(json.dumps(user_profile))
    
    # Define fields that can be marked as unknown
    # Note: We're excluding task instructions as requested
    field_groups = {
        "base_profile": list(new_profile.get("base_profile", {}).keys()),
        "behavioral_traits": list(new_profile.get("behavioral_traits", {}).keys()),
        "contextual_factors": list(new_profile.get("contextual_factors", {}).keys()),
    }
    
    # Handle nested task_specific_attributes separately
    task_specific = new_profile.get("task_profile", {}).get("task_specific_attributes", {})
    field_groups["task_specific"] = []
    
    for key, value in task_specific.items():
        if key != "task" and key != "difficulty_level" and "instructions" not in key:
            if isinstance(value, list):
                # For list values, we'll treat each item as a potential unknown
                for i in range(len(value)):
                    field_groups["task_specific"].append((key, i))
            else:
                field_groups["task_specific"].append(key)
    
    # Calculate total fields and how many to make unknown
    all_fields = []
    for group, fields in field_groups.items():
        for field in fields:
            all_fields.append((group, field))
    
    total_fields = len(all_fields)
    unknown_count = int(total_fields * percentage / 100)
    
    # Randomly select fields to mark as unknown
    unknown_fields = random.sample(all_fields, unknown_count)
    
    # Set selected fields to "unknown"
    for group, field in unknown_fields:
        if group == "base_profile":
            new_profile["base_profile"][field] = "unknown"
        elif group == "behavioral_traits":
            new_profile["behavioral_traits"][field] = "unknown"
        elif group == "contextual_factors":
            new_profile["contextual_factors"][field] = "unknown"
        elif group == "task_specific":
            if isinstance(field, tuple):
                key, index = field
                new_profile["task_profile"]["task_specific_attributes"][key][index] = "unknown"
            else:
                new_profile["task_profile"]["task_specific_attributes"][field] = "unknown"
    
    # Update metadata to reflect the unknown percentage
    if "metadata" in new_profile:
        new_profile["metadata"]["unknown_percentage"] = percentage
    
    return new_profile

if __name__ == "__main__":
    source_directory = "profiles/users/basic"
    generate_unknown_profiles(source_directory)
    print("Profile generation completed.") 