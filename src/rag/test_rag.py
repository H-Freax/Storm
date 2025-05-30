import asyncio
import argparse
import os
import json
from dialogue_rag import DialogueRAG

async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test RAG functionality')
    parser.add_argument('--dialogues-dir', type=str, default='dialogues', help='Dialogues directory')
    parser.add_argument('--query', type=str, required=True, help='Query to search for')
    parser.add_argument('--top-k', type=int, default=3, help='Number of results to retrieve')
    parser.add_argument('--threshold', type=float, default=0.7, help='Similarity threshold')
    args = parser.parse_args()
    
    # Initialize RAG
    rag = DialogueRAG(
        dialogues_dir=args.dialogues_dir,
        top_k=args.top_k,
        similarity_threshold=args.threshold
    )
    
    print(f"Indexing dialogues from {args.dialogues_dir}...")
    rag.index_dialogues()
    
    print(f"\nRetrieving dialogues for query: '{args.query}'")
    retrieved_dialogues = rag.retrieve_relevant_dialogues(args.query)
    
    print(f"\nFound {len(retrieved_dialogues)} relevant dialogue turns:\n")
    if retrieved_dialogues:
        for i, dialogue in enumerate(retrieved_dialogues, 1):
            print(f"Result {i} (Similarity: {dialogue.get('similarity_score', 'N/A'):.4f}):")
            print(f"  User: {dialogue.get('user_message', 'N/A')}")
            print(f"  Assistant: {dialogue.get('assistant_message', 'N/A')}")
            print(f"  Emotional State: {dialogue.get('emotional_state', 'N/A')}")
            print(f"  Intent State: {dialogue.get('intent_state', 'N/A')}")
            print(f"  File: {os.path.basename(dialogue.get('file_path', 'N/A'))}")
            print()
        
        # Format context
        context = rag.format_retrieved_context(retrieved_dialogues)
        print("Formatted Context for LLM:")
        print("-" * 50)
        print(context)
        print("-" * 50)
    else:
        print("No relevant dialogue turns found.")

if __name__ == "__main__":
    asyncio.run(main()) 