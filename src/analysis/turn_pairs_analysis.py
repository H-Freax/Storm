import json
import os
import time
from openai import OpenAI
from typing import Dict, List, Any
from dotenv import load_dotenv
from joblib import Parallel, delayed


# Configuration
load_dotenv(override=True)  # Add override=True to force using .env values
API_KEY = os.getenv("OPENAI_API_KEY")
# print("API Key from env:", API_KEY)
MODEL = "gpt-4o"  # or "gpt-4o"
INPUT_FOLDER = "./example_data/storm_json_final/gpt-4o-mini_with_80"
OUTPUT_FOLDER = "./test/gpt_4o_mini_rag/gpt_4o_mini_rag_turn_analyses/test/gpt_4o_mini_rag"
N_JOBS = 5  # Number of parallel jobs

# Build the prompt for a turn pair
def build_turn_pair_prompt(prev_turn, next_turn, i):
    return f"""
You are given a JSON file representing a multi-turn conversation between a user and an assistant. Each turn includes the user's message, the assistant's response, timestamp, and metadata with satisfaction and inner_thoughts.

For each pair of consecutive turns (e.g., Turn 0 → Turn 1, Turn 1 → Turn 2, etc.), perform the following analysis:

Turn {i} → Turn {i+1}
User Satisfaction
Change from Previous Turn: [Improve / Not Change / Decrease]

Satisfaction Score (X+1): {next_turn['metadata']['hidden_states']['satisfaction']['score']}

Explanation:
Did the assistant's previous response improve the user's experience, keep it steady, or reduce satisfaction? Justify based on the satisfaction score and the user's explanation.

User Clarity
Change in Clarity: [Improve / Not Change / Decrease]

Explanation:
Based on the user's message and inner thoughts in Turn {i + 1}, assess whether their ability to express thoughts, preferences, or goals became clearer, stayed the same, or became less clear. Note specific changes, improvements, or ambiguities.

Now return the result as valid JSON in this exact format:

{{
  "turn_pair": "Turn {i} → Turn {i + 1}",
  "user_satisfaction": {{
    "change": "One of: Improve, Not Change, Decrease",
    "score": {next_turn['metadata']['hidden_states']['satisfaction']['score']},
    "explanation": "Your explanation here"
  }},
  "user_clarity": {{
    "change": "One of: Improve, Not Change, Decrease",
    "explanation": "Your explanation here"
  }}
}}

Here is the conversation snippet:

User Message (Turn {i}): {prev_turn['user_message']}
Assistant Response (Turn {i}): {prev_turn['assistant_message']}

User Message (Turn {i + 1}): {next_turn['user_message']}
Assistant Response (Turn {i + 1}): {next_turn['assistant_message']}

User Inner Thoughts: {next_turn['metadata']['hidden_states']['inner_thoughts']}
Satisfaction Explanation: {next_turn['metadata']['hidden_states']['satisfaction']['explanation']}
""".strip()

# Call OpenAI Chat API
def call_openai(prompt):
    try:
        # Create a new client for each call
        client = OpenAI(api_key=API_KEY)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API call failed: {e}")
        return None

def analyze_turn_pairs_file(input_file: str, output_file: str) -> None:
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        turns = data["turns"]

    results = []

    for i in range(len(turns) - 1):
        prompt = build_turn_pair_prompt(turns[i], turns[i + 1], i)
        print(f"Analyzing {os.path.basename(input_file)} - Turn {i} → Turn {i + 1}...")

        response_text = call_openai(prompt)
        if response_text:
            try:
                # Remove markdown-style JSON fencing if present
                import re
                cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", response_text.strip(), flags=re.IGNORECASE | re.MULTILINE)
                parsed_json = json.loads(cleaned)
                results.append(parsed_json)
            except json.JSONDecodeError:
                print(f"Failed to parse JSON for {os.path.basename(input_file)} - Turn {i} → Turn {i + 1}")
                results.append({
                    "turn_pair": f"Turn {i} → Turn {i + 1}",
                    "error": "Could not parse JSON",
                    "raw_response": response_text
                })
        else:
            results.append({
                "turn_pair": f"Turn {i} - Turn {i + 1}",
                "error": "No response"
            })

        time.sleep(1.2)  # Avoid rate limiting

    with open(output_file, "w", encoding="utf-8") as out:
        json.dump(results, out, indent=4, ensure_ascii=False)

    print(f"Saved analysis to {output_file}")

def process_single_file(filename: str, input_folder: str, output_folder: str) -> None:
    """Process a single file with its own OpenAI client instance."""
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, f"analysis_{filename}")
    analyze_turn_pairs_file(input_path, output_path)

def process_folder(input_folder: str, output_folder: str) -> None:
    """Process all files in the input directory using joblib parallel processing."""
    os.makedirs(output_folder, exist_ok=True)
    
    # Get list of JSON files to process
    json_files = [f for f in os.listdir(input_folder) if f.endswith(".json")]
    
    # Process files in parallel using joblib
    Parallel(n_jobs=N_JOBS)(
        delayed(process_single_file)(filename, input_folder, output_folder)
        for filename in json_files
    )

# Run everything
if __name__ == "__main__":
    process_folder(INPUT_FOLDER, OUTPUT_FOLDER)