import os
import json
import argparse
import time
import datetime
import multiprocessing
import asyncio
import sys
from tqdm import tqdm
from functools import partial
from asymmetric_dialogue_final import AsymmetricDialogueGenerator
from camel.types import ModelType

# Global variable for API key rotation
_openai_api_keys = []
_current_key_index = multiprocessing.Value('i', 0)
_key_lock = multiprocessing.Lock()

def get_next_api_key():
    """Get the next API key in rotation"""
    global _openai_api_keys, _current_key_index, _key_lock
    
    if not _openai_api_keys:
        return os.getenv("OPENAI_API_KEY")
        
    with _key_lock:
        key_index = _current_key_index.value
        _current_key_index.value = (_current_key_index.value + 1) % len(_openai_api_keys)
        
    return _openai_api_keys[key_index]

# Synchronous function for multiprocessing
def generate_single_dialogue_sync(user_profile, num_turns, output_path, rag_enabled=False, 
                                 user_model_type=None, assistant_model_type=None, 
                                 dialogues_dir=None, task_id=None, api_keys=None):
    """Generate a single dialogue in a separate process"""
    user_name = user_profile.get('name', f'unknown-{task_id}')
    timestamp = datetime.datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] Process {os.getpid()} - Task {task_id}: Starting generation for {user_name}")
    start_time = time.time()
    
    try:
        # Set the API keys for rotation in AsymmetricDialogueGenerator
        if api_keys:
            AsymmetricDialogueGenerator.set_api_keys(api_keys)
            print(f"[{timestamp}] Process {os.getpid()} - Task {task_id}: API key rotation enabled")
        
        # Initialize dialogue generator in this process
        generator = AsymmetricDialogueGenerator(
            user_model_type=user_model_type,
            assistant_model_type=assistant_model_type,
            enable_rag=rag_enabled,
            dialogues_dir=dialogues_dir
        )
        
        # Create an event loop and run the async function in it
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        dialogue = loop.run_until_complete(generator.generate_dialogue(
            user_profile=user_profile,
            num_turns=num_turns,
            generate_inner_thoughts=True
        ))
        
        # Save dialogue
        with open(output_path, 'w') as f:
            json.dump(dialogue, f, indent=4)
            
        elapsed = time.time() - start_time
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] Process {os.getpid()} - Task {task_id}: Completed dialogue for {user_name} in {elapsed:.2f} seconds")
        return True, output_path
    except Exception as e:
        elapsed = time.time() - start_time
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        error_msg = f"Error generating dialogue for {user_name} after {elapsed:.2f} seconds: {str(e)}"
        print(f"[{timestamp}] Process {os.getpid()} - Task {task_id}: {error_msg}")
        return False, error_msg

# Keep the async version for backward compatibility
async def generate_single_dialogue(generator, user_profile, num_turns, output_path, semaphore, task_id, rag_enabled=False):
    """Generate a single dialogue with semaphore control (legacy async version)"""
    user_name = user_profile.get('name', f'unknown-{task_id}')
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Task {task_id}: Starting generation for {user_name}")
    start_time = time.time()
    async with semaphore:
        try:
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Task {task_id}: Acquired semaphore for {user_name}")
            # Generate dialogue
            dialogue = await generator.generate_dialogue(
                user_profile=user_profile,
                num_turns=num_turns,
                generate_inner_thoughts=True
            )
            
            # Save dialogue
            with open(output_path, 'w') as f:
                json.dump(dialogue, f, indent=4)
                
            elapsed = time.time() - start_time
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Task {task_id}: Completed dialogue for {user_name} in {elapsed:.2f} seconds")
            return True, output_path
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = f"Error generating dialogue for {user_name} after {elapsed:.2f} seconds: {str(e)}"
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Task {task_id}: {error_msg}")
            return False, error_msg

def batch_generate_dialogues_mp():
    """Generate multiple dialogues in parallel using multiprocessing"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Batch generate dialogues with parallel processing')
    parser.add_argument('--rag', action='store_true', help='Enable RAG mode')
    parser.add_argument('--dialogues-dir', type=str, default='dialogues', help='Dialogues directory for RAG')
    parser.add_argument('--user-model', type=str, default='GPT_4O_MINI', help='User model type')
    parser.add_argument('--assistant-model', type=str, default='GPT_4O_MINI', help='Assistant model type')
    parser.add_argument('--num-turns', type=int, default=15, help='Number of dialogue turns')
    parser.add_argument('--profiles-dir', type=str, default='profiles/users/basic', help='User profiles directory')
    parser.add_argument('--max-concurrent', type=int, default=10, help='Maximum number of concurrent processes')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--api-keys', type=str, default=None, help='JSON array of OpenAI API keys for rotation')
    parser.add_argument('--share-profile', action='store_true', help='Share user profile information with assistant')
    args = parser.parse_args()
    
    print(f"Starting batch generation with max_concurrent={args.max_concurrent}")
    
    # Error tracking
    critical_errors = 0
    max_critical_errors = 3
    
    # Set up API keys for rotation
    api_keys = []
    if args.api_keys:
        try:
            api_keys = json.loads(args.api_keys)
            print(f"Using API key rotation with {len(api_keys)} keys")
        except json.JSONDecodeError:
            print("Warning: Failed to parse API keys JSON. Using default API key.")
    
    # Convert model type strings to ModelType enums if they match enum names
    user_model_type = args.user_model
    assistant_model_type = args.assistant_model
    
    print(f"Using user model: {user_model_type}")
    print(f"Using assistant model: {assistant_model_type}")
    
    # Check for known model compatibility issues
    incompatible_models = ["qwen/qwen3-4b:free", "qwen/qwen3-235b-a22b", "mistralai/mistral-medium-3"]
    if assistant_model_type in incompatible_models:
        print(f"⚠️ WARNING: Model {assistant_model_type} has known compatibility issues with the CAMEL library")
        print("You may encounter errors during generation. Consider using a different model.")
    
    # Set number of turns for dialogue
    num_turns = args.num_turns
    
    # Create output directory structure
    rag_suffix = "_rag" if args.rag else ""
    base_dir = args.dialogues_dir
    
    # Get assistant model name
    if hasattr(assistant_model_type, 'name'):
        assistant_model_name = assistant_model_type.name.lower()
    else:
        assistant_model_name = str(assistant_model_type).replace("/", "_")  # Replace slashes in model paths
    
    model_dir = f"{base_dir}/{assistant_model_name}{rag_suffix}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Collect all profiles
    profiles = []
    output_paths = []
    
    profiles_dir = args.profiles_dir
    
    # Check if directory exists
    if not os.path.exists(profiles_dir):
        print(f"Error: Profiles directory '{profiles_dir}' does not exist!")
        sys.exit(1)
        
    profile_files = os.listdir(profiles_dir)
    if not profile_files:
        print(f"Error: No files found in profiles directory '{profiles_dir}'!")
        sys.exit(1)
        
    for filename in profile_files:
        if filename.endswith(".json"):
            try:
                # Load user profile
                with open(os.path.join(profiles_dir, filename), 'r') as f:
                    user_profile = json.load(f)
                
                # Prepare output path
                rag_indicator = "with_rag" if args.rag else "without_rag"
                output_filename = filename.replace("_user_", f"_dialogue_{num_turns}turns_{rag_indicator}_")
                output_path = os.path.join(model_dir, output_filename)
                
                profiles.append(user_profile)
                output_paths.append(output_path)
                
            except Exception as e:
                print(f"Error loading profile {filename}: {str(e)}")
    
    if not profiles:
        print(f"Error: No valid profiles found in '{profiles_dir}'!")
        sys.exit(1)
        
    print(f"Found {len(profiles)} user profiles to process")
    print(f"Max concurrent processes: {args.max_concurrent}")
    
    # Create a process pool with the specified concurrency limit
    start_time = time.time()
    success_count = 0
    results = []
    
    # Create list of tasks
    tasks = []
    for i, (user_profile, output_path) in enumerate(zip(profiles, output_paths)):
        tasks.append({
            'user_profile': user_profile,
            'output_path': output_path,
            'task_id': i+1
        })
    
    # Execute tasks with progress bar
    with multiprocessing.Pool(processes=min(args.max_concurrent, len(tasks))) as pool:
        with tqdm(total=len(tasks), desc="Generating dialogues") as pbar:
            # Define the function to call for each task
            generate_fn = partial(
                generate_single_dialogue_sync,
                num_turns=num_turns,
                rag_enabled=args.rag,
                user_model_type=user_model_type,
                assistant_model_type=assistant_model_type,
                dialogues_dir=args.dialogues_dir,
                api_keys=api_keys
            )
            
            # Submit all tasks
            for task_id, (user_profile, output_path) in enumerate(zip(profiles, output_paths), 1):
                result = pool.apply_async(
                    generate_fn, 
                    kwds={
                        'user_profile': user_profile,
                        'output_path': output_path,
                        'task_id': task_id
                    },
                    callback=lambda _: pbar.update(1)
                )
                results.append(result)
            
            # Wait for all tasks to complete
            pool.close()
            
            if args.debug:
                while not all(result.ready() for result in results):
                    active_count = sum(1 for result in results if not result.ready())
                    timestamp = datetime.datetime.now().strftime('%H:%M:%S')
                    print(f"[{timestamp}] Currently {active_count} tasks in progress")
                    time.sleep(5)
            
            pool.join()
    
    # Process results
    for result in results:
        success, error_info = result.get()
        if success:
            success_count += 1
        else:
            # Check for critical errors - NoneType is not iterable typically indicates model incompatibility
            if 'NoneType' in error_info and 'not iterable' in error_info:
                critical_errors += 1
                print(f"⚠️ Critical error detected: {error_info}")
                if critical_errors >= max_critical_errors:
                    print(f"❌ Too many critical errors ({critical_errors}). Model may be incompatible.")
                    break
    
    total_time = time.time() - start_time
    timestamp = datetime.datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] Successfully generated {success_count} out of {len(tasks)} dialogues")
    print(f"Total execution time: {total_time:.2f} seconds")
    
    if len(tasks) > 0:
        print(f"Average time per dialogue: {total_time/len(tasks):.2f} seconds")
        print(f"Throughput: {len(tasks)/total_time:.2f} dialogues per second")
    
    # Exit with error code if too many failures
    if success_count == 0 or critical_errors >= max_critical_errors:
        print("Exiting with error code due to critical failures")
        sys.exit(1)
        
    return results

# Rename original function for backward compatibility
async def batch_generate_dialogues_async():
    """Original asyncio version of batch_generate_dialogues"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Batch generate dialogues with parallel processing')
    parser.add_argument('--rag', action='store_true', help='Enable RAG mode')
    parser.add_argument('--dialogues-dir', type=str, default='dialogues', help='Dialogues directory for RAG')
    parser.add_argument('--user-model', type=str, default='GPT_4O_MINI', help='User model type')
    parser.add_argument('--assistant-model', type=str, default='GPT_4O_MINI', help='Assistant model type')
    parser.add_argument('--num-turns', type=int, default=15, help='Number of dialogue turns')
    parser.add_argument('--profiles-dir', type=str, default='profiles/users/basic', help='User profiles directory')
    parser.add_argument('--max-concurrent', type=int, default=10, help='Maximum number of concurrent dialogues')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--api-keys', type=str, default=None, help='JSON array of OpenAI API keys for rotation')
    parser.add_argument('--share-profile', action='store_true', help='Share user profile information with assistant')
    args = parser.parse_args()
    
    print(f"Starting batch generation with max_concurrent={args.max_concurrent}")
    
    # Convert model type strings to ModelType enums if they match enum names
    try:
        user_model_type = getattr(ModelType, args.user_model)
    except AttributeError:
        user_model_type = args.user_model
        
    try:
        assistant_model_type = getattr(ModelType, args.assistant_model)
    except AttributeError:
        assistant_model_type = args.assistant_model

    # Initialize dialogue generator with optional RAG
    generator = AsymmetricDialogueGenerator(
        user_model_type=user_model_type,
        assistant_model_type=assistant_model_type,
        enable_rag=args.rag,
        dialogues_dir=args.dialogues_dir,
        share_profile_with_assistant=args.share_profile
    )
    
    # Set number of turns for dialogue
    num_turns = args.num_turns
    
    # Create output directory structure
    rag_suffix = "_rag" if args.rag else ""
    base_dir = args.dialogues_dir
    
    # Handle different model type formats (enum or string path)
    assistant_model_name = generator.assistant_model.model_type.name.lower() if isinstance(generator.assistant_model.model_type, ModelType) else str(generator.assistant_model.model_type)
    assistant_model_name = assistant_model_name.replace("/", "_")  # Replace slashes in model paths
    
    model_dir = f"{base_dir}/{assistant_model_name}{rag_suffix}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Create semaphore for controlling concurrency
    import asyncio
    semaphore = asyncio.Semaphore(args.max_concurrent)
    
    # Collect all profiles
    profiles = []
    output_paths = []
    
    profiles_dir = args.profiles_dir
    
    # Check if directory exists
    if not os.path.exists(profiles_dir):
        print(f"Error: Profiles directory '{profiles_dir}' does not exist!")
        return []
        
    profile_files = os.listdir(profiles_dir)
    if not profile_files:
        print(f"Error: No files found in profiles directory '{profiles_dir}'!")
        return []
        
    for filename in profile_files:
        if filename.endswith(".json"):
            try:
                # Load user profile
                with open(os.path.join(profiles_dir, filename), 'r') as f:
                    user_profile = json.load(f)
                
                # Prepare output path
                rag_indicator = "with_rag" if args.rag else "without_rag"
                output_filename = filename.replace("_user_", f"_dialogue_{num_turns}turns_{rag_indicator}_")
                output_path = os.path.join(model_dir, output_filename)
                
                profiles.append(user_profile)
                output_paths.append(output_path)
                
            except Exception as e:
                print(f"Error loading profile {filename}: {str(e)}")
    
    if not profiles:
        print(f"Error: No valid profiles found in '{profiles_dir}'!")
        return []
        
    print(f"Found {len(profiles)} user profiles to process")
    print(f"Max concurrent tasks: {args.max_concurrent}")
    
    # Create tasks for all profiles
    tasks = []
    for i, (user_profile, output_path) in enumerate(zip(profiles, output_paths)):
        task = generate_single_dialogue(
            generator, 
            user_profile, 
            num_turns, 
            output_path, 
            semaphore,
            i+1,  # task_id 
            args.rag
        )
        tasks.append(task)
    
    # Execute tasks with progress bar
    results = []
    success_count = 0
    
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Starting {len(tasks)} tasks execution with {args.max_concurrent} concurrent tasks...")
    progress_bar = tqdm(total=len(tasks), desc="Generating dialogues")
    
    # Start all tasks immediately
    start_time = time.time()
    in_progress_tasks = set()
    
    # Debug active coroutines if debug flag is set
    if args.debug:
        async def monitor_tasks():
            while in_progress_tasks:
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Currently {len(in_progress_tasks)} tasks in progress")
                await asyncio.sleep(5)
        
        monitor = asyncio.create_task(monitor_tasks())
    
    for task in tasks:
        in_progress_tasks.add(task)
    
    for future in asyncio.as_completed(tasks):
        success, result = await future
        in_progress_tasks.discard(future)
        
        if success:
            success_count += 1
        else:
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {result}")  # Print error message
        
        results.append((success, result))
        progress_bar.update(1)
    
    if args.debug and 'monitor' in locals():
        monitor.cancel()
        
    total_time = time.time() - start_time
    progress_bar.close()
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Successfully generated {success_count} out of {len(tasks)} dialogues")
    print(f"Total execution time: {total_time:.2f} seconds")
    
    if len(tasks) > 0:
        print(f"Average time per dialogue: {total_time/len(tasks):.2f} seconds")
        print(f"Throughput: {len(tasks)/total_time:.2f} dialogues per second")
    
    return results

# Make original function name point to new function for backward compatibility
batch_generate_dialogues = batch_generate_dialogues_mp

if __name__ == "__main__":
    try:
        batch_generate_dialogues_mp()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        print(f"Batch generation failed: {str(e)}")
        sys.exit(1)  # Exit with error code
    sys.exit(0)  # Success