import os
import json
import argparse
from asymmetric_dialogue_final import AsymmetricDialogueGenerator
from camel.types import ModelType
from joblib import Parallel, delayed

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate dialogues with optional RAG')
    parser.add_argument('--rag', action='store_true', help='Enable RAG mode')
    parser.add_argument('--dialogues-dir', type=str, default='dialogues', help='Dialogues directory for RAG')
    parser.add_argument('--user-model', type=str, default='GPT_4O_MINI', help='User model type')
    parser.add_argument('--assistant-model', type=str, default='GPT_4O_MINI', help='Assistant model type')
    parser.add_argument('--num-turns', type=int, default=15, help='Number of dialogue turns')
    parser.add_argument('--profiles-dir', type=str, default='profiles/users/basic', help='User profiles directory')
    parser.add_argument('--storage-dir', type=str, default='dialogue_vectors', help='Vector storage directory')
    parser.add_argument('--vector-storage', type=str, default='qdrant', choices=['qdrant', 'milvus'], help='Vector storage type')
    parser.add_argument('--threshold', type=float, default=0.7, help='Similarity threshold')
    parser.add_argument('--share-profile', action='store_true', help='Share user profile information with assistant')
    args = parser.parse_args()
    
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
        storage_dir=args.storage_dir,
        vector_storage_type=args.vector_storage,
        similarity_threshold=args.threshold,
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
    
    # Process all JSON files in the profiles directory using joblib
    profiles_dir = args.profiles_dir
    json_files = [f for f in os.listdir(profiles_dir) if f.endswith(".json")]

    def process_file(filename):
        try:
            with open(os.path.join(profiles_dir, filename), 'r') as f:
                user_profile = json.load(f)

            dialogue = generator.generate_dialogue(
                user_profile=user_profile,
                num_turns=num_turns,
                generate_inner_thoughts=True
            )

            rag_indicator = "with_rag" if args.rag else "without_rag"
            output_filename = filename.replace("_user_", f"_dialogue_{num_turns}turns_{rag_indicator}_")
            output_path = os.path.join(model_dir, output_filename)

            with open(output_path, 'w') as f:
                json.dump(dialogue, f, indent=4)

            print(f"Generated {num_turns}-turn dialogue for {filename} {'with' if args.rag else 'without'} RAG")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

    Parallel(n_jobs=10, backend="threading")(  
        delayed(process_file)(filename) for filename in json_files
    )

if __name__ == "__main__":
    main() 