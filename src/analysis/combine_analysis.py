import json
import os
import sys
from pathlib import Path

def combine_files(dialogue_file, intent_file, analysis_file, output_dir="combined_analysis"):
    """
    Combine dialogue, intent clarity, and turn analysis files into a single JSON file.
    
    Args:
        dialogue_file: Path to the dialogue JSON file
        intent_file: Path to the intent clarity JSON file
        analysis_file: Path to the turn analysis JSON file
        output_dir: Directory to save the combined file
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load dialogue file
    with open(dialogue_file, 'r', encoding='utf-8') as f:
        dialogue_data = json.load(f)
    
    # Load intent clarity file
    with open(intent_file, 'r', encoding='utf-8') as f:
        intent_data = json.load(f)
    
    # Load analysis file
    with open(analysis_file, 'r', encoding='utf-8') as f:
        analysis_data = json.load(f)
    
    # Extract base filename without path and extension
    base_filename = os.path.basename(dialogue_file)
    base_name = os.path.splitext(base_filename)[0]
    
    # Create combined structure
    combined_data = {
        "dialogue": dialogue_data,
        "intent_clarity": intent_data,
        "turn_analysis": analysis_data,
        "metadata": {
            "dialogue_file": dialogue_file,
            "intent_file": intent_file,
            "analysis_file": analysis_file,
            "created_at": None  # Will be filled in when saved
        }
    }
    
    # Add summary information
    combined_data["summary"] = {
        "total_turns": len(dialogue_data["turns"]),
        "final_satisfaction_score": dialogue_data["metadata"]["final_hidden_states"]["satisfaction"]["score"],
        "final_satisfaction_explanation": dialogue_data["metadata"]["final_hidden_states"]["satisfaction"]["explanation"],
        "task": dialogue_data["metadata"]["user_profile"]["task_profile"]["task"] if "user_profile" in dialogue_data["metadata"] else None,
        "model": dialogue_data["metadata"]["models"]["assistant_model"] if "models" in dialogue_data["metadata"] else None
    }
    
    # Save combined file
    output_file = os.path.join(output_dir, f"combined_{base_name}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        # Add timestamp before saving
        from datetime import datetime
        combined_data["metadata"]["created_at"] = datetime.now().isoformat()
        json.dump(combined_data, f, indent=2)
    
    print(f"Combined file saved to: {output_file}")
    return output_file

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python combine_analysis.py <dialogue_file> <intent_file> <analysis_file> [output_dir]")
        sys.exit(1)
    
    dialogue_file = sys.argv[1]
    intent_file = sys.argv[2]
    analysis_file = sys.argv[3]
    
    output_dir = "combined_analysis"
    if len(sys.argv) > 4:
        output_dir = sys.argv[4]
    
    combine_files(dialogue_file, intent_file, analysis_file, output_dir) 