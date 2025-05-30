import os
import json
import argparse
import asyncio
import time
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from dialogue_rag import DialogueRAG
from camel.storages import QdrantStorage, MilvusStorage
from camel.embeddings import OpenAIEmbedding

def load_env_file():
    """Load environment variables from .env file"""
    # Get the project root directory (3 levels up from this file)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    env_path = os.path.join(project_root, '.env')
    
    print(f"\nDebug: Looking for .env file at: {env_path}")
    
    if not os.path.exists(env_path):
        print("Error: .env file not found")
        raise FileNotFoundError("No .env file found in project root")
    
    print("Debug: Found .env file, reading contents...")
    
    # Read and set environment variables
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    # Remove any whitespace or newlines
                    value = value.strip()
                    os.environ[key] = value
                    if key == 'OPENAI_API_KEY':
                        print(f"Debug: Loaded {key}={value[:10]}...")
                        # Validate API key format
                        if not value.startswith('sk-'):
                            raise ValueError("Invalid OpenAI API key format. Key should start with 'sk-'")
                except Exception as e:
                    print(f"Debug: Error parsing line: {line}")
                    print(f"Debug: Error details: {str(e)}")
                    raise

# Load environment variables at module level
load_env_file()

# Verify OPENAI_API_KEY is set
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("Error: OPENAI_API_KEY not found in .env file")
    raise ValueError("OPENAI_API_KEY environment variable is not set")
else:
    print(f"\nDebug: OPENAI_API_KEY found: {api_key[:10]}...")
    # Additional validation
    if not api_key.startswith('sk-'):
        raise ValueError("Invalid OpenAI API key format. Key should start with 'sk-'")

class JsonToRAGImporter:
    """
    A class to import dialogue JSON files to RAG storage with progress tracking.
    """
    
    def __init__(
        self,
        dialogues_dir: str = "dialogues",
        storage_dir: str = "dialogue_vectors",
        collection_name: str = "dialogue_data",
        vector_storage_type: str = "qdrant",
        similarity_threshold: float = 0.75,
        embedding_model: Optional[Any] = None,
    ):
        """
        Initialize the JSON to RAG importer.
        
        Args:
            dialogues_dir: Directory containing dialogue JSON files
            storage_dir: Directory to store vector database
            collection_name: Name of the vector collection
            vector_storage_type: Type of vector storage ('qdrant' or 'milvus')
            similarity_threshold: Similarity threshold for retrieval
            embedding_model: Custom embedding model (optional)
        """
        self.dialogues_dir = dialogues_dir
        self.storage_dir = storage_dir
        self.collection_name = collection_name
        self.vector_storage_type = vector_storage_type.lower()
        self.similarity_threshold = similarity_threshold
        
        # Initialize embedding model if not provided
        if embedding_model is None:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            self.embedding_model = OpenAIEmbedding()
        else:
            self.embedding_model = embedding_model
            
        # Initialize vector storage based on type
        self._init_vector_storage()
        
        # Initialize RAG
        self.rag = DialogueRAG(
            dialogues_dir=dialogues_dir,
            embedding_model=self.embedding_model,
            vector_storage=self.vector_storage,
            similarity_threshold=similarity_threshold
        )
        
    def _init_vector_storage(self):
        """Initialize vector storage based on specified type"""
        vector_dim = self.embedding_model.get_output_dim()
        
        # Ensure storage directory exists and has proper permissions
        os.makedirs(self.storage_dir, exist_ok=True)
        os.chmod(self.storage_dir, 0o755)
        
        if self.vector_storage_type == "qdrant":
            try:
                import qdrant_client
                import sqlite3
                
                # Clean up any existing lock files
                lock_file = os.path.join(self.storage_dir, ".lock")
                if os.path.exists(lock_file):
                    os.remove(lock_file)
                
                # Configure SQLite for higher limits
                db_path = os.path.join(self.storage_dir, "collection", self.collection_name, "storage.sqlite")
                if os.path.exists(db_path):
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    # Set higher limits
                    cursor.execute("PRAGMA max_page_count = 2147483646")  # ~2TB
                    cursor.execute("PRAGMA page_size = 4096")
                    cursor.execute("PRAGMA journal_mode = WAL")  # Use Write-Ahead Logging
                    cursor.execute("PRAGMA synchronous = NORMAL")  # Faster writes
                    cursor.execute("PRAGMA cache_size = -2000")  # 2MB cache
                    cursor.execute("PRAGMA temp_store = MEMORY")  # Store temp tables in memory
                    conn.commit()
                    conn.close()
                    
                self.vector_storage = QdrantStorage(
                    vector_dim=vector_dim,
                    path=self.storage_dir,
                    collection_name=self.collection_name
                )
            except ImportError:
                print("Error: qdrant_client not installed. Installing with pip...")
                os.system("pip install qdrant-client")
                print("Retrying initialization...")
                import qdrant_client
                self.vector_storage = QdrantStorage(
                    vector_dim=vector_dim,
                    path=self.storage_dir,
                    collection_name=self.collection_name
                )
        elif self.vector_storage_type == "milvus":
            try:
                import pymilvus
                self.vector_storage = MilvusStorage(
                    vector_dim=vector_dim,
                    url_and_api_key=(
                        os.path.join(self.storage_dir, "milvus.db"),
                        ""
                    ),
                    collection_name=self.collection_name
                )
            except ImportError:
                print("Error: pymilvus not installed. Installing with pip...")
                os.system("pip install pymilvus")
                print("Retrying initialization...")
                import pymilvus
                self.vector_storage = MilvusStorage(
                    vector_dim=vector_dim,
                    url_and_api_key=(
                        os.path.join(self.storage_dir, "milvus.db"),
                        ""
                    ),
                    collection_name=self.collection_name
                )
        else:
            raise ValueError(f"Unsupported vector storage type: {self.vector_storage_type}")
    
    def find_dialogue_files(self) -> List[str]:
        """Find all dialogue JSON files in the dialogues directory"""
        dialogue_files = []
        for root, _, files in os.walk(self.dialogues_dir):
            for file in files:
                if file.endswith('.json'):
                    dialogue_files.append(os.path.join(root, file))
        return dialogue_files
    
    def import_dialogues(self, batch_size: int = 10) -> bool:
        """
        Import dialogues from JSON files to RAG storage with progress bar.
        
        Args:
            batch_size: Number of files to process in a batch for the progress bar
            
        Returns:
            bool: True if import succeeded, False otherwise
        """
        dialogue_files = self.find_dialogue_files()
        if not dialogue_files:
            print(f"No dialogue files found in {self.dialogues_dir}")
            return False
        
        print(f"\nStarting dialogue import from {self.dialogues_dir} to {self.vector_storage_type} storage ({self.storage_dir})...")
        print(f"Found {len(dialogue_files)} files to process")
        
        # Process files in batches with progress bar
        imported_turns = 0
        total_turns = 0
        
        # Create progress bar for files
        with tqdm(total=len(dialogue_files), desc="Processing files", position=0) as pbar_files:
            for i in range(0, len(dialogue_files), batch_size):
                batch_files = dialogue_files[i:i+batch_size]
                
                for file_path in batch_files:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            dialogue = json.load(f)
                        
                        # Count turns for this file
                        file_turn_count = len(dialogue.get("turns", []))
                        total_turns += file_turn_count
                        turn_count = 0
                        
                        # Process turns in the dialogue
                        for turn in dialogue.get("turns", []):
                            # Skip turns with no user message or assistant message
                            if not turn.get("user_message") or not turn.get("assistant_message"):
                                continue
                            
                            # Create content for indexing
                            turn_content = {
                                "file_path": file_path,
                                "turn_number": turn["turn_number"],
                                "user_message": turn["user_message"],
                                "assistant_message": turn["assistant_message"],
                                "user_profile": dialogue["metadata"]["user_profile"],
                                "emotional_state": turn["metadata"]["hidden_states"]["emotional_state"],
                                "intent_state": turn["metadata"]["hidden_states"]["intent_state"],
                            }
                            
                            # Convert to text format for embedding
                            text_content = f"""File: {file_path}
Turn: {turn["turn_number"]}
User: {turn["user_message"]}
Assistant: {turn["assistant_message"]}
Emotional State: {turn["metadata"]["hidden_states"]["emotional_state"]}
Intent State: {turn["metadata"]["hidden_states"]["intent_state"]}
"""
                            
                            # Add custom metadata to content
                            metadata = {
                                "file_path": file_path,
                                "turn_number": turn["turn_number"],
                                "dialogue_data": json.dumps(turn_content)
                            }
                            
                            # Process the content
                            try:
                                self.rag.process_text(
                                    content=text_content,
                                    metadata=metadata
                                )
                                imported_turns += 1
                                turn_count += 1
                            except Exception as e:
                                print(f"\nError processing turn {turn['turn_number']} from {file_path}: {str(e)}")
                        
                        # Update progress bar with current file info
                        pbar_files.set_postfix({
                            'current_file': os.path.basename(file_path),
                            'turns_imported': f"{turn_count}/{file_turn_count}"
                        })
                        pbar_files.update(1)
                        
                    except Exception as e:
                        print(f"\nError processing file {file_path}: {str(e)}")
                        pbar_files.update(1)
        
        print(f"\nImport completed:")
        print(f"- Total files processed: {len(dialogue_files)}")
        print(f"- Total turns imported: {imported_turns}")
        print(f"- Total turns found: {total_turns}")
        self.rag.is_indexed = True
        return True
    
    def test_query(self, query: str, top_k: int = 3) -> None:
        """
        Test a query against the imported RAG.
        
        Args:
            query: The query text
            top_k: Number of results to retrieve
        """
        if not self.rag.is_indexed:
            print("No data indexed yet. Run import_dialogues first.")
            return
        
        print(f"Testing query: '{query}'")
        start_time = time.time()
        retrieved_dialogues = self.rag.retrieve_relevant_dialogues(query)
        elapsed_time = time.time() - start_time
        
        print(f"Retrieved {len(retrieved_dialogues)} results in {elapsed_time:.2f} seconds:\n")
        
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
            context = self.rag.format_retrieved_context(retrieved_dialogues)
            print("Formatted Context for LLM:")
            print("-" * 50)
            print(context)
            print("-" * 50)
        else:
            print("No relevant dialogue turns found.")

async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Import dialogue JSON files to RAG storage')
    parser.add_argument('--dialogues-dir', type=str, default='dialogues', help='Dialogues directory')
    parser.add_argument('--storage-dir', type=str, default='dialogue_vectors', help='Vector storage directory')
    parser.add_argument('--collection-name', type=str, default='dialogue_data', help='Vector collection name')
    parser.add_argument('--vector-storage', type=str, default='qdrant', choices=['qdrant', 'milvus'], help='Vector storage type')
    parser.add_argument('--threshold', type=float, default=0.7, help='Similarity threshold')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for processing')
    parser.add_argument('--query', type=str, help='Test query after import')
    parser.add_argument('--top-k', type=int, default=3, help='Number of results for test query')
    args = parser.parse_args()
    
    try:
        print("\nInitializing RAG importer...")
        # Initialize importer
        importer = JsonToRAGImporter(
            dialogues_dir=args.dialogues_dir,
            storage_dir=args.storage_dir,
            collection_name=args.collection_name,
            vector_storage_type=args.vector_storage,
            similarity_threshold=args.threshold
        )
        
        # Import dialogues with progress bar
        print("\nStarting dialogue import process...")
        success = importer.import_dialogues(batch_size=args.batch_size)
        
        if success:
            print("\nImport completed successfully!")
            if args.query:
                # Test query if provided
                print("\nTesting query...")
                importer.test_query(args.query, top_k=args.top_k)
        else:
            print("\nImport failed!")
            return
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    asyncio.run(main()) 