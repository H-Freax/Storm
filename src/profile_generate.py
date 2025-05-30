import os
import asyncio
import json
from datetime import datetime
from profile_generators.user_profile_generator import UserProfileGenerator
from profile_generators.task_profile_generator import TaskProfileGenerator

def check_existing_profiles(category: str, task: str) -> tuple[bool, bool]:
    """
    Check if profiles already exist for a given category and task.
    
    Args:
        category: The category name
        task: The task name
        
    Returns:
        tuple: (task_profiles_exist, user_profiles_exist)
    """
    task_pattern = f"profiles/tasks/basic/{category}_{task.replace(' ', '_')}_option_"
    user_pattern = f"profiles/users/basic/{category}_{task.replace(' ', '_')}_user_"
    
    # Check task profiles
    task_profiles_exist = any(
        f.startswith(task_pattern) and f.endswith('.json')
        for f in os.listdir("profiles/tasks/basic")
    )
    
    # Check user profiles
    user_profiles_exist = any(
        f.startswith(user_pattern) and f.endswith('.json')
        for f in os.listdir("profiles/users/basic")
    )
    
    return task_profiles_exist, user_profiles_exist

async def main():
    # Create directories for profiles
    os.makedirs("profiles/users/basic", exist_ok=True)
    os.makedirs("profiles/tasks/basic", exist_ok=True)

    # Initialize profile generators
    task_profile_generator = TaskProfileGenerator()
    user_profile_generator = UserProfileGenerator()

    # Define task categories and their specific tasks
    task_categories = {
        "technology": [
            "buy a smartphone",
            "reset an online password",
            "teach my parent to use video calls"
        ],
        "healthcare": [
            "refill my prescription",
            "schedule a doctor visit",
            "find a caregiver for an elderly person"
        ],
        "daily living": [
            "order groceries online",
            "set medication reminders",
            "arrange transportation to a clinic"
        ],
        "housing": [
            "rent an apartment",
            "find an accessible home",
            "arrange home modifications for elderly"
        ],
        "caregiver support": [
            "book a nurse for my father",
            "choose a phone for my mom",
            "find cognitive exercises for dementia prevention"
        ]
    }

    # Generate task profiles for each category and task
    for category, tasks in task_categories.items():
        for task in tasks:
            # Check if profiles already exist
            task_profiles_exist, user_profiles_exist = check_existing_profiles(category, task)
            
            if task_profiles_exist:
                print(f"Skipping task profiles generation for {category} - {task} (already exist)")
            else:
                try:
                    # Generate multiple options for each task
                    task_profiles = await task_profile_generator.generate_task_options(
                        task=task,
                        num_options=5,
                        difficulty_variation=True,
                        include_uncertain=True
                    )

                    # Save each task profile
                    for i, profile in enumerate(task_profiles):
                        filename = f"profiles/tasks/basic/{category}_{task.replace(' ', '_')}_option_{i+1}.json"
                        task_profile_generator.save_task_profile(profile, filename)
                        print(f"Saved task profile: {filename}")

                except Exception as e:
                    print(f"Error generating task profiles for {task}: {str(e)}")

            # Generate random user profiles for each task
            if user_profiles_exist:
                print(f"Skipping user profiles generation for {category} - {task} (already exist)")
            else:
                for i in range(10):  # Generate 3 user profiles for each task
                    try:
                        # Generate user profile with task context
                        user_profile = await user_profile_generator.generate_profile(
                            task=task,
                            difficulty_level=(i % 5) + 1  # Vary difficulty level
                        )
                        
                        # Save the user profile
                        filename = f"profiles/users/basic/{category}_{task.replace(' ', '_')}_user_{i+1}.json"
                        user_profile_generator.save_profile(user_profile, filename)
                        print(f"Saved user profile: {filename}")

                    except Exception as e:
                        print(f"Error generating user profile for {task} (attempt {i+1}): {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())