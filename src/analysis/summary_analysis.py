#!/usr/bin/env python3
import json
import os
import time
from openai import OpenAI
from typing import List, Dict, Any
from dotenv import load_dotenv
from joblib import Parallel, delayed

# Configuration
load_dotenv(override=True)  # force using .env values
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o"
INPUT_FOLDER = "./example_data/storm_json_final/gpt-4o-mini_with_80"
OUTPUT_FOLDER = "./example_data/storm_json_final_summary/llama-3.3-70b-instruct_with_80_summary"
N_JOBS = 5  # number of parallel jobs


def build_summary_prompt(turns: List[Dict[str, Any]], filename: str) -> str:
    """
    Build a prompt to generate a detailed summary analysis of a multi-turn conversation.
    """
    # Assemble conversation overview
    conversation_text = ""
    for idx, turn in enumerate(turns):
        user_msg = turn.get("user_message", "").replace('\n', ' ')
        score = turn.get("metadata", {}).get("hidden_states", {}).get("satisfaction", {}).get("score", "N/A")
        conversation_text += f"Turn {idx} - User: {user_msg}\n"  
        conversation_text += f"Satisfaction score: {score}\n\n"

    prompt = f"""
You are given a multi-turn conversation between a user and an assistant. Each turn includes a user satisfaction score.

Consider that each user's background, expertise, and goals may vary; present your analysis as nuanced insights and generalizable recommendations, avoiding absolute judgments.

Generate a comprehensive, detailed summary analysis of the conversation. Return strictly valid JSON with these fields:
1. summary_overall: A concise evaluation of overall user satisfaction trend (e.g., positive, negative, mixed).
2. topics_covered: A list of key topics or user intents addressed throughout the conversation.
3. statistics: An object containing:
   - average_score: Average satisfaction score across all turns.
   - min_score: Minimum score observed.
   - max_score: Maximum score observed.
   - score_variance: Variance of the satisfaction scores.
4. satisfaction_evolution: A list of objects for each turn:
   - turn_index: Index of the turn.
   - score: Satisfaction score at that turn.
   - delta: Change in score from the previous turn (null for first turn).
5. important_turns: A list of objects identifying critical turns where satisfaction changes significantly (e.g., change >= 2):
   - turn_index: Index of the user turn.
   - user_message: The user's message at that turn.
   - score_before: Score at the previous turn.
   - score_after: Score at the following turn.
   - change: Numeric difference (score_after - score_before).
   - reason: Explanation based on conversation content.
6. detailed_findings: A list of objects providing deep insights for each important turn:
   - turn_index: Index of the turn.
   - context_before: The assistant and user messages immediately before this turn.
   - context_after: The assistant and user messages immediately after this turn.
   - analysis: Detailed rationale for why the score changed.
   - recommendation: Suggestions for how the assistant could improve at this point.
7. contextual_notes: A list of any relevant context, caveats, or user metadata considerations that influenced the analysis.
8. general_insights: A list of general patterns or best practices inferred from this conversation that could apply to a broad range of users.

Conversation file: {filename}

{conversation_text}
""".strip()
    return prompt


def call_openai(prompt: str) -> str:
    """
    Call the OpenAI ChatCompletion API and return the assistant response.
    """
    try:
        client = OpenAI(api_key=API_KEY)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API call failed: {e}")
        return ""


def analyze_summary_file(input_file: str, output_file: str) -> None:
    """
    Analyze a single conversation JSON file and write the summary JSON.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    turns = data.get("turns", [])
    print(f"Analyzing summary for {os.path.basename(input_file)}")

    prompt = build_summary_prompt(turns, os.path.basename(input_file))
    response = call_openai(prompt)

    # Clean fences if present
    import re
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.strip(), flags=re.IGNORECASE | re.MULTILINE)

    try:
        parsed = json.loads(cleaned)
        with open(output_file, 'w', encoding='utf-8') as out:
            json.dump(parsed, out, indent=4, ensure_ascii=False)
    except json.JSONDecodeError:
        with open(output_file, 'w', encoding='utf-8') as out:
            json.dump({
                "error": "Failed to parse JSON",
                "raw_response": response
            }, out, indent=4, ensure_ascii=False)


def process_single_file(filename: str, input_folder: str, output_folder: str) -> None:
    """
    Process one file: generate summary and write to output.
    """
    input_path = os.path.join(input_folder, filename)
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"summary_{filename}")
    analyze_summary_file(input_path, output_path)


def process_folder(input_folder: str, output_folder: str) -> None:
    """
    Process all JSON files in a folder in parallel.
    """
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.endswith('.json')]

    Parallel(n_jobs=N_JOBS)(
        delayed(process_single_file)(filename, input_folder, output_folder)
        for filename in files
    )


if __name__ == '__main__':
    process_folder(INPUT_FOLDER, OUTPUT_FOLDER) 