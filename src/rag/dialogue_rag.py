import os
import json
import logging
import glob
from typing import Dict, Any, List, Optional, Tuple, Union
import re
import numpy as np
import hashlib
import time
from joblib import Parallel, delayed
from tqdm import tqdm

from camel.storages import QdrantStorage, MilvusStorage
from camel.storages.vectordb_storages.base import VectorRecord
from camel.retrievers import VectorRetriever
from camel.embeddings import OpenAIEmbedding

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def process_single_text(embedding_model, content: str, metadata: Dict[str, Any]) -> Tuple[str, List[float], Dict[str, Any]]:
    """Process a single text content and return its embedding and metadata."""
    try:
        # Generate embedding
        embedding = embedding_model.embed(content)
        
        # Create unique ID
        content_id = hashlib.md5((content + str(metadata)).encode('utf-8')).hexdigest()
        
        return content_id, embedding, metadata
    except Exception as e:
        logger.error(f"Error processing single text: {str(e)}")
        raise

class DialogueRAG:
    """
    A class that handles Retrieval-Augmented Generation (RAG) using dialogue data.
    This class loads dialogue data from JSON files, indexes them for retrieval,
    and enhances dialogue generation with related historical dialogues.
    """

    def __init__(self, 
                 dialogues_dir: str = "dialogues",
                 embedding_model: Optional[Any] = None,
                 vector_storage: Optional[Any] = None,
                 similarity_threshold: float = 0.75,
                 top_k: int = 3):
        """
        Initialize the DialogueRAG with the specified parameters.
        
        Args:
            dialogues_dir: Directory containing dialogue JSON files
            embedding_model: The embedding model to use (default: OpenAIEmbedding)
            vector_storage: The vector storage to use (default: QdrantStorage)
            similarity_threshold: Minimum similarity score for retrieval
            top_k: Maximum number of results to retrieve
        """
        self.dialogues_dir = dialogues_dir
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        
        # Initialize embedding model if not provided
        if embedding_model is None:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            self.embedding_model = OpenAIEmbedding()
        else:
            self.embedding_model = embedding_model
        
        # Initialize vector storage if not provided
        if vector_storage is None:
            try:
                # Try to import qdrant_client
                import qdrant_client
                self.vector_storage = QdrantStorage(
                    vector_dim=self.embedding_model.get_output_dim(),
                    path="dialogue_vectors",
                    collection_name="dialogue_data"
                )
            except ImportError:
                logger.warning("qdrant_client not installed, falling back to simple in-memory storage")
                # Fallback to simple in-memory storage
                self.vector_storage = SimpleVectorStorage(
                    vector_dim=self.embedding_model.get_output_dim()
                )
        else:
            self.vector_storage = vector_storage
        
        # Initialize retriever
        self.retriever = VectorRetriever(
            embedding_model=self.embedding_model,
            storage=self.vector_storage
        )
        
        # Track if dialogues have been indexed
        self.is_indexed = False
    
    def process_text(self, content: str, metadata: Dict[str, Any] = None) -> None:
        """
        Process a text content directly, embed it and store it in the vector database.
        
        Args:
            content: Text content to process
            metadata: Optional metadata to store with the content
        """
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                # Generate embedding
                embedding = self.embedding_model.embed(content)
                
                # Create unique ID using content hash
                content_id = hashlib.md5((content + str(metadata or {})).encode('utf-8')).hexdigest()
                
                # Store in vector database
                if metadata is None:
                    metadata = {}
                    
                # Create VectorRecord for Qdrant
                record = VectorRecord(
                    id=content_id,
                    vector=embedding,
                    payload=metadata
                )
                
                # Clean up any existing lock files before operation
                storage_dir = getattr(self.vector_storage, 'path', None)
                if storage_dir:
                    lock_file = os.path.join(storage_dir, ".lock")
                    if os.path.exists(lock_file):
                        os.remove(lock_file)
                
                # Try add_vectors first
                if hasattr(self.vector_storage, 'add_vectors'):
                    try:
                        self.vector_storage.add_vectors(
                            vectors=[embedding],
                            metadatas=[metadata],
                            ids=[content_id]
                        )
                        return  # Success, exit the function
                    except Exception as e:
                        logger.error(f"Error in add_vectors (attempt {attempt + 1}/{max_retries}): {str(e)}")
                
                # Fall back to add method
                if hasattr(self.vector_storage, 'add'):
                    try:
                        # Ensure we're not in a transaction
                        if hasattr(self.vector_storage, 'rollback'):
                            self.vector_storage.rollback()
                        
                        self.vector_storage.add(records=[record])
                        return  # Success, exit the function
                    except Exception as e:
                        logger.error(f"Error in add (attempt {attempt + 1}/{max_retries}): {str(e)}")
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            continue
                else:
                    logger.error("Vector storage does not support adding vectors")
                    return
                    
            except Exception as e:
                logger.error(f"Error processing text (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    raise
    
    def process_text_batch(self, contents: List[str], metadatas: List[Dict[str, Any]] = None) -> None:
        """
        Process multiple text contents in batch using parallel processing.
        
        Args:
            contents: List of text contents to process
            metadatas: Optional list of metadata to store with the contents
        """
        if not contents:
            return
            
        if metadatas is None:
            metadatas = [{}] * len(contents)
        elif len(metadatas) != len(contents):
            raise ValueError("Number of metadatas must match number of contents")
            
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                # Process texts in parallel
                n_jobs = min(4, len(contents))  # Use up to 4 parallel jobs
                results = Parallel(n_jobs=n_jobs)(
                    delayed(process_single_text)(self.embedding_model, content, metadata)
                    for content, metadata in zip(contents, metadatas)
                )
                
                # Unpack results
                content_ids, embeddings, processed_metadatas = zip(*results)
                
                # Create VectorRecords
                records = [
                    VectorRecord(
                        id=content_id,
                        vector=embedding,
                        payload=metadata
                    )
                    for content_id, embedding, metadata in zip(content_ids, embeddings, processed_metadatas)
                ]
                
                # Clean up any existing lock files before operation
                storage_dir = getattr(self.vector_storage, 'path', None)
                if storage_dir:
                    lock_file = os.path.join(storage_dir, ".lock")
                    if os.path.exists(lock_file):
                        os.remove(lock_file)
                
                # Try add_vectors first
                if hasattr(self.vector_storage, 'add_vectors'):
                    try:
                        self.vector_storage.add_vectors(
                            vectors=list(embeddings),
                            metadatas=list(processed_metadatas),
                            ids=list(content_ids)
                        )
                        return  # Success, exit the function
                    except Exception as e:
                        logger.error(f"Error in add_vectors batch (attempt {attempt + 1}/{max_retries}): {str(e)}")
                
                # Fall back to add method
                if hasattr(self.vector_storage, 'add'):
                    try:
                        # Ensure we're not in a transaction
                        if hasattr(self.vector_storage, 'rollback'):
                            self.vector_storage.rollback()
                        
                        self.vector_storage.add(records=records)
                        return  # Success, exit the function
                    except Exception as e:
                        logger.error(f"Error in add batch (attempt {attempt + 1}/{max_retries}): {str(e)}")
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            continue
                else:
                    logger.error("Vector storage does not support adding vectors")
                    return
                    
            except Exception as e:
                logger.error(f"Error processing text batch (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    raise

    def index_dialogues(self) -> None:
        """
        Index all dialogues in the dialogues directory.
        This creates embeddings for each dialogue turn and stores them for retrieval.
        """
        if self.is_indexed:
            logger.info("Dialogues already indexed")
            return
        
        try:
            # Get all dialogue files
            dialogue_files = []
            for root, _, files in os.walk(self.dialogues_dir):
                for file in files:
                    if file.endswith('.json'):
                        dialogue_files.append(os.path.join(root, file))
            
            if not dialogue_files:
                logger.warning(f"No dialogue files found in {self.dialogues_dir}")
                return
            
            logger.info(f"Indexing {len(dialogue_files)} dialogue files")
            
            # Process each dialogue file
            batch_size = 16  # Process 10 turns at a time
            for file_path in tqdm(dialogue_files, desc="Processing files"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        dialogue = json.load(f)
                    
                    # Collect turns for batch processing
                    batch_contents = []
                    batch_metadatas = []
                    
                    # Process turns in the dialogue
                    for turn in dialogue["turns"]:
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
                        
                        batch_contents.append(text_content)
                        batch_metadatas.append(metadata)
                        
                        # Process batch when it reaches batch_size
                        if len(batch_contents) >= batch_size:
                            self.process_text_batch(batch_contents, batch_metadatas)
                            batch_contents = []
                            batch_metadatas = []
                    
                    # Process remaining items in the last batch
                    if batch_contents:
                        self.process_text_batch(batch_contents, batch_metadatas)
                    
                except Exception as e:
                    logger.error(f"Error processing dialogue file {file_path}: {str(e)}")
            
            self.is_indexed = True
            logger.info("Dialogue indexing complete")
            
        except Exception as e:
            logger.error(f"Error indexing dialogues: {str(e)}")
            raise
    
    def retrieve_relevant_dialogues(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant dialogues based on the query.
        
        Args:
            query: The query to search for
            
        Returns:
            List of relevant dialogue turns
        """
        if not self.is_indexed:
            self.index_dialogues()
        
        try:
            # Validate query input
            if not query or not isinstance(query, str):
                logger.warning("Invalid query input. Query must be a non-empty string.    "+str(type(query))+"    "+query)
                return []
            
            # Clean and normalize query
            query = query.strip()
            if not query:
                logger.warning("Empty query after cleaning.")
                return []
            
            # Generate embedding for the query
            try:
                query_embedding = self.embedding_model.embed(query)
            except Exception as e:
                logger.error(f"Error generating embedding for query: {str(e)}")
                return []

            # Use the correct retrieval method depending on storage type
            if hasattr(self.vector_storage, 'query'):
                # QdrantStorage or similar
                from camel.storages.vectordb_storages import VectorDBQuery
                query_obj = VectorDBQuery(
                    query_vector=query_embedding,
                    top_k=self.top_k
                )
                results = self.vector_storage.query(query=query_obj, filter_conditions=None)
                # Convert to expected result format
                processed_results = []
                for result in results:
                    try:
                        metadata = result.record.payload or {}
                        dialogue_data = json.loads(metadata.get("dialogue_data", "{}"))
                        dialogue_data["similarity_score"] = result.similarity
                        processed_results.append(dialogue_data)
                    except (json.JSONDecodeError, AttributeError) as e:
                        logger.error(f"Error processing retrieval result: {str(e)}")
                return processed_results
            else:
                # Fallback for SimpleVectorStorage
                results = self.vector_storage.similarity_search(
                    query_embedding=query_embedding,
                    k=self.top_k,
                    similarity_threshold=self.similarity_threshold
                )
                processed_results = []
                for result in results:
                    try:
                        metadata = result.get("metadata", {})
                        dialogue_data = json.loads(metadata.get("dialogue_data", "{}"))
                        dialogue_data["similarity_score"] = result.get("score", 0)
                        processed_results.append(dialogue_data)
                    except (json.JSONDecodeError, AttributeError) as e:
                        logger.error(f"Error processing retrieval result: {str(e)}")
                return processed_results
        except Exception as e:
            logger.error(f"Error retrieving dialogues: {str(e)}")
            return []

    def format_retrieved_context(self, retrieved_dialogues: List[Dict[str, Any]]) -> str:
        """
        Format retrieved dialogues into a context string for the LLM.
        
        Args:
            retrieved_dialogues: List of retrieved dialogue turns
            
        Returns:
            Formatted context string
        """
        if not retrieved_dialogues:
            return ""
        
        context = "### Retrieved similar conversations:\n\n"
        for i, dialogue in enumerate(retrieved_dialogues, 1):
            context += f"Conversation {i}:\n"
            context += f"User: {dialogue.get('user_message', 'N/A')}\n"
            context += f"Assistant: {dialogue.get('assistant_message', 'N/A')}\n"
            context += f"Emotional State: {dialogue.get('emotional_state', 'N/A')}\n"
            context += f"Intent State: {dialogue.get('intent_state', 'N/A')}\n\n"
        
        return context


class SimpleVectorStorage:
    """
    A simple in-memory vector storage for testing purposes.
    This class provides a fallback when vector database libraries are not available.
    """
    
    def __init__(self, vector_dim: int):
        """
        Initialize the simple vector storage.
        
        Args:
            vector_dim: Dimension of the vectors
        """
        self.vector_dim = vector_dim
        self.vectors = []
        self.metadatas = []
        self.ids = []
    
    def add(self, vector: List[float], metadata: Dict[str, Any], id: str) -> None:
        """
        Add a vector to the storage.
        
        Args:
            vector: Vector to add
            metadata: Metadata for the vector
            id: ID for the vector
        """
        self.vectors.append(np.array(vector))
        self.metadatas.append(metadata)
        self.ids.append(id)
    
    def add_vectors(self, vectors: List[List[float]], metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
        """
        Add multiple vectors to the storage.
        
        Args:
            vectors: List of vectors to add
            metadatas: List of metadata for the vectors
            ids: List of IDs for the vectors
        """
        for vector, metadata, id in zip(vectors, metadatas, ids):
            self.add(vector, metadata, id)
    
    def similarity_search(self, query_embedding: List[float], k: int = 3, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of results
        """
        if not self.vectors:
            return []
        
        query_vector = np.array(query_embedding)
        
        # Calculate cosine similarity
        similarities = []
        for i, vector in enumerate(self.vectors):
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            if similarity >= similarity_threshold:
                similarities.append((i, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        results = []
        for i, similarity in similarities[:k]:
            results.append({
                "id": self.ids[i],
                "metadata": self.metadatas[i],
                "score": similarity
            })
        
        return results 