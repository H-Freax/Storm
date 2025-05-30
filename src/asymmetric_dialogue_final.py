import os
import sys
from typing import Dict, Any, List, Optional,Union
import asyncio
import dotenv
from dotenv import load_dotenv
import json
import logging
from datetime import datetime
import traceback
from camel.agents import ChatAgent
from camel.types import ModelType,ModelPlatformType
from camel.models import ModelFactory
from camel.configs import ChatGPTConfig, OpenRouterConfig
import re
from rag.dialogue_rag import DialogueRAG
from camel.embeddings import OpenAIEmbedding
import time
import random
import threading

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class AsymmetricDialogueGenerator:
    """
    A class for generating asymmetric dialogues between a user and an assistant.
    This class handles the creation of natural conversations with different difficulty levels,
    profile management, and state tracking.
    """

    # Configuration for message length constraints
    MESSAGE_LENGTHS = {
        "user": {
            "min": 20,  # Minimum length for user messages
            "max": 100,  # Maximum length for user messages
            "default": 50  # Default target length for user messages
        },
        "assistant": {
            "min": 30,  # Minimum length for assistant messages
            "max": 150,  # Maximum length for assistant messages
            "default": 80  # Default target length for assistant messages
        }
    }

    # Class-level key rotation variables
    _openai_api_keys = []
    _current_key_index = 0
    _key_lock = threading.Lock()

    @classmethod
    def set_api_keys(cls, keys):
        """Set the API keys for rotation"""
        if keys and isinstance(keys, list):
            cls._openai_api_keys = keys
            cls._current_key_index = 0
            logger.info(f"API key rotation enabled with {len(keys)} keys")

    @classmethod
    def get_next_api_key(cls, key_type="openai"):
        """Get the next API key in rotation"""
        if key_type.lower() != "openai" or not cls._openai_api_keys:
            # For non-OpenAI models or if no keys are set for rotation
            return os.getenv(f"{key_type.upper()}_API_KEY")
        
        with cls._key_lock:
            key_index = cls._current_key_index
            cls._current_key_index = (cls._current_key_index + 1) % len(cls._openai_api_keys)
            
        return cls._openai_api_keys[key_index]

    def __init__(self, 
                 user_model_type: Union[ModelType, str] = ModelType.GPT_4O_MINI,
                 assistant_model_type: Union[ModelType, str] = ModelType.GPT_4O_MINI,
                 enable_rag: bool = False,
                 dialogues_dir: str = "dialogues",
                 vector_storage_type: str = "qdrant",
                 similarity_threshold: float = 0.75,
                 storage_dir: str = "dialogue_vectors",
                 share_profile_with_assistant: bool = False):
        """
        Initialize the asymmetric dialogue generator with separate models for user and assistant.
        
        Args:
            user_model_type: The model type to use for generating user responses
            assistant_model_type: The model type to use for generating assistant responses
            enable_rag: Whether to enable Retrieval-Augmented Generation
            dialogues_dir: Directory containing dialogue JSON files for RAG
            vector_storage_type: Type of vector storage ('qdrant' or 'milvus')
            similarity_threshold: Minimum similarity score for retrieval
            storage_dir: Directory to store vector database for RAG (default: 'dialogue_vectors')
            share_profile_with_assistant: Whether to share user profile information with assistant
        """
        # Model configurations
        self.user_model_config = {
            "temperature": 1,
            "max_tokens": 10000,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.1
        }
        self.assistant_model_config = {
            "temperature": 0,
            "max_tokens": 10000,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.1
        }
        
        try:
            user_model_type = getattr(ModelType, user_model_type)
        except AttributeError:
            user_model_type = user_model_type
            
        try:
            assistant_model_type = getattr(ModelType, assistant_model_type)
        except AttributeError:
            assistant_model_type = assistant_model_type
            
        # RAG configuration
        self.enable_rag = enable_rag
        self.rag = None
        self.storage_dir = storage_dir
        
        # Profile sharing setting
        self.share_profile_with_assistant = share_profile_with_assistant
        
        # API request configuration
        self.max_retries = 5  # Maximum number of retries for API calls
        self.base_delay = 1.0  # Base delay for exponential backoff (seconds)
        self.max_delay = 32.0  # Maximum delay for exponential backoff (seconds)
        
        try:
            # Initialize models
            self.user_model = self._create_model(user_model_type, self.user_model_config)
            self.assistant_model = self._create_model(assistant_model_type, self.assistant_model_config)
            logger.info("Successfully initialized asymmetric dialogue models")
            
            # Initialize RAG if enabled
            if self.enable_rag:
                try:
                    self.rag = DialogueRAG(
                        dialogues_dir=dialogues_dir,
                        similarity_threshold=similarity_threshold
                    )
                    logger.info(f"Successfully initialized RAG with dialogues from {dialogues_dir}")
                except Exception as e:
                    logger.error(f"Failed to initialize RAG: {str(e)}")
                    logger.warning("Running without RAG due to initialization error")
                    self.enable_rag = False
                    self.rag = None

        except Exception as e:
            error_msg = f"Error initializing asymmetric dialogue models: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            raise

    def _get_api_key(self, model_type: Union[ModelType, str]) -> str:
        """Get the appropriate API key based on model type with rotation support."""
        if isinstance(model_type, ModelType):
            return self.get_next_api_key("openai")
        else:
            return self.get_next_api_key("openrouter")

    def _create_model(self, model_type: Union[ModelType, str], config: Dict[str, Any]) -> Any:
        """Create a model instance with the given configuration."""
        api_key = self._get_api_key(model_type)

        if isinstance(model_type, ModelType):
            model_config = ChatGPTConfig(**config)
            platform = ModelPlatformType.OPENAI
        else:
            model_config = OpenRouterConfig(**config)
            platform = ModelPlatformType.OPENROUTER

        return ModelFactory.create(
            model_platform=platform,
            model_type=model_type,
            model_config_dict=model_config.as_dict(),
            api_key=api_key,
            timeout=240.0  # 2-minute timeout for API calls
        )

    async def _call_model_with_retry(self, agent: ChatAgent, prompt: str) -> str:
        """
        Call the model with exponential backoff retry strategy.
        
        Args:
            agent: The ChatAgent to use for the call
            prompt: The prompt to send to the model
            
        Returns:
            str: Model response content
        """
        retry_count = 0
        last_exception = None
        
        while retry_count < self.max_retries:
            try:
                # Attempt to get response from model
                response = agent.step(prompt)
                content = self._extract_message_content(response)
                
                # Check if content is empty or very short (likely an error)
                if not content or len(content) < 5:
                    raise ValueError(f"Empty or very short response received: '{content}'")
                    
                return content
                
            except Exception as e:
                last_exception = e
                retry_count += 1
                
                if retry_count >= self.max_retries:
                    logger.warning(f"Max retries ({self.max_retries}) reached. Using fallback response.")
                    break
                
                # Calculate exponential backoff delay with jitter
                delay = min(self.base_delay * (2 ** (retry_count - 1)) + random.uniform(0, 0.5), self.max_delay)
                logger.warning(f"API call failed (attempt {retry_count}/{self.max_retries}): {str(e)}. Retrying in {delay:.2f}s")
                await asyncio.sleep(delay)
        
        # If all attempts failed, log the error and return a fallback response
        logger.error(f"All {self.max_retries} attempts failed: {str(last_exception)}")
        
        # Generate different fallback responses based on agent type
        is_user_agent = agent.model == self.user_model
        if is_user_agent:
            fallback_responses = [
                "I'm still thinking about this...",
                "Hmm, that's interesting.",
                "Let me consider that for a moment.",
                "I'm not sure how to respond to that.",
                "Could you tell me more about that?"
            ]
        else:
            fallback_responses = [
                "I apologize, but I'm having trouble processing your request right now. Could you please rephrase your question?",
                "There seems to be a technical issue. Can you try asking your question in a different way?",
                "I'm sorry for the inconvenience. Let's try to approach this from a different angle.",
                "I apologize for the delay. Could you provide more details about what you're looking for?",
                "I'm having difficulty connecting to my knowledge base. Could you please clarify your question?"
            ]
        
        return random.choice(fallback_responses)

    def _extract_message_content(self, message) -> str:
        """
        Extract content from various response types.
        
        Args:
            message: The response object from the model
            
        Returns:
            str: The extracted content
        """
        try:
            if message is None:
                logger.warning("Received None response from model")
                return "I'm sorry, I couldn't process your request. Could you try again?"
            
            if hasattr(message, 'msg') and hasattr(message.msg, 'content'):
                return message.msg.content
            elif hasattr(message, 'content'):
                return message.content
            elif hasattr(message, 'msgs') and message.msgs:
                return message.msgs[-1].content
            else:
                return str(message)
        except Exception as e:
            logger.error(f"Error extracting message content: {str(e)}")
            return str(message)

    def _extract_inner_thoughts(self, content: str) -> str:
        """
        Extract inner thoughts from the content if present.
        
        Args:
            content: Message content to extract inner thoughts from
            
        Returns:
            str: Extracted inner thoughts or empty string if not found
        """
        try:
            # Handle format: [INNER_THOUGHTS: thoughts]
            if "[INNER_THOUGHTS:" in content and "]" in content:
                start = content.find("[INNER_THOUGHTS:") + len("[INNER_THOUGHTS:")
                end = content.find("]", start)
                return content[start:end].strip()
            
            # Handle format: [INNER_THOUGHTS] thoughts [/INNER_THOUGHTS]
            elif "[INNER_THOUGHTS]" in content and "[/INNER_THOUGHTS]" in content:
                start = content.find("[INNER_THOUGHTS]") + len("[INNER_THOUGHTS]")
                end = content.find("[/INNER_THOUGHTS]")
                return content[start:end].strip()
            
            return ""
        except Exception as e:
            logger.error(f"Error extracting inner thoughts: {str(e)}")
            return ""

    def _clean_content(self, content: str) -> str:
        """
        Remove inner thoughts and satisfaction information from the content.
        This method removes any content between [INNER_THOUGHTS] and [SATISFACTION] tags
        to get the clean message content.
        
        Args:
            content: Message content to clean
            
        Returns:
            str: Clean message content without inner thoughts and satisfaction
        """
        try:
            # First, identify and preserve the user message if it exists after [SATISFACTION]
            user_message = ""
            if "[SATISFACTION]" in content and "[/SATISFACTION]" in content:
                sat_end = content.find("[/SATISFACTION]") + len("[/SATISFACTION]")
                if sat_end < len(content):
                    user_message = content[sat_end:].strip()
                    logger.debug(f"Found user message after SATISFACTION: {user_message}")

            # Remove inner thoughts section (format: [INNER_THOUGHTS: thoughts])
            if "[INNER_THOUGHTS:" in content and "]" in content:
                start = content.find("[INNER_THOUGHTS:")
                end = content.find("]", start) + 1
                content = content[:start] + content[end:].strip()
                logger.debug("Removed INNER_THOUGHTS: format")
            
            # Remove inner thoughts section (format: [INNER_THOUGHTS] thoughts [/INNER_THOUGHTS])
            elif "[INNER_THOUGHTS]" in content and "[/INNER_THOUGHTS]" in content:
                start = content.find("[INNER_THOUGHTS]")
                end = content.find("[/INNER_THOUGHTS]") + len("[/INNER_THOUGHTS]")
                content = content[:start] + content[end:].strip()
                logger.debug("Removed INNER_THOUGHTS tag format")
            
            # Remove satisfaction section
            if "[SATISFACTION]" in content and "[/SATISFACTION]" in content:
                start = content.find("[SATISFACTION]")
                end = content.find("[/SATISFACTION]") + len("[/SATISFACTION]")
                content = content[:start] + content[end:].strip()
                logger.debug("Removed SATISFACTION section")
            
            # Clean up extra whitespace and newlines
            content = re.sub(r'\n\s*\n', '\n', content)  # Remove multiple newlines
            content = content.strip()  # Remove leading/trailing whitespace
            
            # If we have a user message, use it as the final content
            if user_message:
                logger.debug(f"Returning user message: {user_message}")
                return user_message
            
            logger.debug(f"Returning cleaned content: {content}")
            return content
        except Exception as e:
            logger.error(f"Error cleaning content: {str(e)}")
            return content

    def _determine_emotional_state(self, content: str, previous_state: str = "neutral") -> str:
        """
        Determine the emotional state based on the visible message content and previous state.
        
        Args:
            content: Message content to analyze
            previous_state: Previous emotional state for continuity
            
        Returns:
            str: Current emotional state
        """
        content_lower = content.lower()
        
        # Emotional keywords mapping
        emotional_keywords = {
            "happy": ["happy", "excited", "great", "wonderful", "perfect", "love", "like", "joy", "pleased", "delighted", "thrilled", "glad", "enjoying", "satisfied", "positive"],
            "frustrated": ["frustrated", "annoyed", "upset", "angry", "disappointed", "not happy", "irritated", "bothered", "fed up", "aggravated", "displeased", "impatient", "agitated", "exasperated"],
            "confused": ["confused", "not sure", "don't understand", "unclear", "complicated", "puzzled", "perplexed", "lost", "unsure", "bewildered", "disoriented", "uncertain", "ambiguous"],
            "interested": ["interesting", "tell me more", "could you explain", "how does", "intrigued", "curious", "fascinated", "engaged", "captivated", "keen", "eager", "want to know"],
            "skeptical": ["really?", "are you sure", "is that true", "not convinced", "doubtful", "suspicious", "unconvinced", "questioning", "dubious", "disbelieving", "hard to believe"],
            "neutral": ["okay", "alright", "fine", "good", "yes", "no", "sure", "maybe", "possibly", "perhaps", "hmm", "i see", "understood", "noted"],
            "anxious": ["worried", "nervous", "anxious", "concerned", "uneasy", "apprehensive", "stressed", "tense", "troubled", "afraid", "fearful", "panicked", "alarmed"],
            "grateful": ["thank you", "thanks", "appreciate", "grateful", "thankful", "indebted", "obliged", "appreciative", "recognition", "acknowledging", "gratitude"],
            "surprised": ["wow", "oh", "really", "surprising", "unexpected", "shocked", "amazed", "astonished", "startled", "stunned", "taken aback", "incredible", "unbelievable"],
            "disappointed": ["disappointed", "letdown", "shame", "too bad", "unfortunate", "regret", "unsatisfactory", "dismayed", "disheartened", "unfulfilled", "discontented"],
            "hopeful": ["hope", "looking forward", "anticipate", "optimistic", "excited about", "expecting", "anticipated", "promising", "encouraging", "reassuring", "positive outlook"]
        }
        
        # Count emotional keywords
        emotion_scores = {emotion: 0 for emotion in emotional_keywords}
        for emotion, keywords in emotional_keywords.items():
            for keyword in keywords:
                if keyword in content_lower:
                    emotion_scores[emotion] += 1
        
        # Get the emotion with highest score
        max_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        
        # If no clear emotion detected, maintain previous state
        if max_emotion[1] == 0:
            return previous_state
            
        return max_emotion[0]

    def _determine_intent_state(self, content: str, previous_state: str = "exploring") -> str:
        """
        Determine the intent state based on the visible message content and previous state.
        
        Args:
            content: Message content to analyze
            previous_state: Previous intent state for continuity
            
        Returns:
            str: Current intent state
        """
        content_lower = content.lower()
        
        # Intent keywords mapping
        intent_keywords = {
            "exploring": ["looking for", "interested in", "tell me about", "what are", "show me", "find", "search for", "discover", "learn about", "explain", "describe", "overview of", "information on", "curious about"],
            "comparing": ["difference between", "which is better", "compare", "versus", "vs", "pros and cons", "advantages of", "disadvantages of", "similarities", "contrasting", "how does it compare", "better choice", "alternatives to"],
            "deciding": ["should I", "which one", "recommend", "suggestion", "advise", "what would you choose", "best option", "worth it", "good choice", "help me decide", "make a decision", "right for me", "considering"],
            "confirming": ["are you sure", "is that right", "does it have", "can it", "verify", "confirm", "is it true", "really", "actually", "definitely", "guarantee", "promise", "certain", "double-check"],
            "purchasing": ["how much", "price", "buy", "purchase", "cost", "ordering", "payment", "discount", "sale", "shipping", "availability", "in stock", "checkout", "add to cart", "where can I get"],
            "leaving": ["thank you", "goodbye", "bye", "see you", "thanks", "appreciate it", "that's all", "ending", "finished", "done", "chat later", "signing off", "talk later"],
            "troubleshooting": ["problem", "issue", "not working", "error", "fix", "help me with", "troubleshoot", "broken", "stuck", "won't work", "doesn't work", "failed", "bugs", "glitches"],
            "requesting": ["can you", "could you", "please", "would you", "need you to", "want you to", "help me", "assist me", "I'd like you to", "request", "favor"],
            "expressing_satisfaction": ["great", "awesome", "perfect", "excellent", "wonderful", "love it", "satisfied", "happy with", "good job", "well done", "thanks", "appreciate"],
            "expressing_dissatisfaction": ["disappointed", "unhappy", "not satisfied", "didn't work", "not good", "terrible", "awful", "frustrated", "upset", "not what I wanted", "dislike"],
            "inquiring": ["how do I", "how to", "steps to", "guide for", "tutorial", "instructions", "process of", "way to", "method for", "approach to"],
            "clarifying": ["what do you mean", "don't understand", "confused", "unclear", "elaborate", "explain more", "clarify", "be more specific", "meaning of", "rephrase"]
        }
        
        # Count intent keywords
        intent_scores = {intent: 0 for intent in intent_keywords}
        for intent, keywords in intent_keywords.items():
            for keyword in keywords:
                if keyword in content_lower:
                    intent_scores[intent] += 1
        
        # Get the intent with highest score
        max_intent = max(intent_scores.items(), key=lambda x: x[1])
        
        # If no clear intent detected, maintain previous state
        if max_intent[1] == 0:
            return previous_state
            
        return max_intent[0]
    
    def _determine_inner_intent_state(self, inner_thoughts: str, previous_state: str = "exploring") -> str:
        """
        Determine the user's real intent based on their inner thoughts and previous state.
        This captures intentions that might not be explicitly stated in the visible message.
        
        Args:
            inner_thoughts: Inner thoughts content to analyze
            previous_state: Previous inner intent state for continuity
            
        Returns:
            str: Current inner intent state
        """
        if not inner_thoughts:
            return previous_state
            
        inner_thoughts_lower = inner_thoughts.lower()
        
        # Inner intent keywords mapping - more focused on true intentions
        inner_intent_keywords = {
            "exploring": ["need information", "want to know", "curious", "just browsing", "researching", "gathering info", "learning", "understand", "figure out", "not sure yet", "looking into"],
            "comparing": ["weighing options", "pros and cons", "better choice", "similarities", "differences", "alternatives", "compare", "contrast", "evaluation", "weigh", "prefer", "which one is better"],
            "deciding": ["almost ready", "need to decide", "make up my mind", "making a choice", "leaning towards", "considering", "thinking about getting", "might choose", "on the fence", "close to deciding"],
            "confirming": ["double-check", "verify", "make sure", "confirm", "reassurance", "validate", "certain", "correct information", "trust but verify", "need proof", "skeptical"],
            "purchasing": ["ready to buy", "want to purchase", "where to buy", "looking to get", "willing to pay", "budget", "cost concerns", "spend money", "deal", "bargain", "checkout"],
            "leaving": ["need to go", "end this", "wrap up", "moving on", "done here", "finished", "that's all I needed", "got what I came for", "time to leave", "goodbye"],
            "resisting": ["not telling everything", "hiding my real goal", "being vague on purpose", "not revealing", "keeping cards close", "holding back", "secretly want", "actual intention", "real reason"],
            "testing": ["testing their knowledge", "seeing if they know", "checking competence", "pushing to see response", "challenging", "probing", "testing limits", "seeing if capable"],
            "manipulating": ["get them to", "convince them", "make them think", "lead them to believe", "appear as if", "trick", "misdirection", "real agenda", "hidden motive", "strategic"],
            "distrusting": ["don't believe", "skeptical", "not sure I trust", "dubious", "suspicious", "questionable", "doubt", "can't trust", "not convinced", "wary of", "hesitant"],
            "regretting": ["should have asked", "forgot to mention", "didn't say", "wish I had", "too late now", "missed opportunity", "should have been clearer", "miscommunicated", "not what I meant"],
            "hesitating": ["nervous about", "afraid to ask", "hesitant", "uncertain", "reluctant", "apprehensive", "can't decide", "overthinking", "worried", "anxious", "reservations"]
        }
        
        # Count inner intent keywords
        inner_intent_scores = {intent: 0 for intent in inner_intent_keywords}
        for intent, keywords in inner_intent_keywords.items():
            for keyword in keywords:
                if keyword in inner_thoughts_lower:
                    inner_intent_scores[intent] += 1
        
        # Get the intent with highest score
        max_inner_intent = max(inner_intent_scores.items(), key=lambda x: x[1])
        
        # If no clear intent detected, maintain previous state
        if max_inner_intent[1] == 0:
            return previous_state
            
        return max_inner_intent[0]

    def _create_user_prompt(self, user_profile: Dict[str, Any]) -> str:
        """
        Create a prompt for the user LLM based on the user profile.
        
        Args:
            user_profile: Dictionary containing user profile information
            
        Returns:
            str: Formatted prompt for the user LLM
        """
        # Get profile information
        name = user_profile.get('name', 'User')
        description = user_profile.get('description', '')
        base_profile = user_profile.get('base_profile', {})
        behavioral_traits = user_profile.get('behavioral_traits', {})
        contextual_factors = user_profile.get('contextual_factors', {})
        task_profile = user_profile.get('task_profile', {})
        example_dimensions = user_profile.get('example_dimensions', {})

        # Build base profile section
        base_profile_section = "Your base profile (private):\n"
        for key, value in base_profile.items():
            base_profile_section += f"- {key}: {value}\n"

        # Build behavioral traits section
        behavioral_traits_section = "\nYour behavioral traits (private):\n"
        for key, value in behavioral_traits.items():
            behavioral_traits_section += f"- {key}: {value}\n"

        # Build contextual factors section
        contextual_factors_section = "\nYour contextual factors (private):\n"
        for key, value in contextual_factors.items():
            contextual_factors_section += f"- {key}: {value}\n"

        # Build task profile section
        task_profile_section = "\nYour task profile (private):\n"
        task_profile_section += f"- Task: {task_profile.get('task', '')}\n"
        task_profile_section += f"- Difficulty Level: {task_profile.get('difficulty_level', 1)}\n"
        
        # Add task-specific attributes
        task_specific = task_profile.get('task_specific_attributes', {})
        if task_specific:
            task_profile_section += "\nTask-specific attributes:\n"
            for key, value in task_specific.items():
                if isinstance(value, list):
                    task_profile_section += f"- {key}: {', '.join(value)}\n"
                else:
                    task_profile_section += f"- {key}: {value}\n"

        # Get difficulty instructions
        instructions = task_profile.get('instructions', {})
        dialogue_instruction = instructions.get('dialogue', '')
        profile_instruction = instructions.get('profile', '')
        hidden_state_instruction = instructions.get('hidden_state', '')

        # Get example dimensions
        examples = example_dimensions.get('examples', [])
        example_section = "\nExample messages:\n"
        for i, example in enumerate(examples, 1):
            example_section += f"{i}. {example}\n"

        # Format the complete prompt
        prompt = f"""You are {name}. {description}

{base_profile_section}
{behavioral_traits_section}
{contextual_factors_section}
{task_profile_section}

Difficulty Instructions:
- Dialogue: {dialogue_instruction}
- Profile: {profile_instruction}
- Hidden State: {hidden_state_instruction}

{example_section}

Message Format Requirements:
1. Your messages should be between {self.MESSAGE_LENGTHS['user']['min']} and {self.MESSAGE_LENGTHS['user']['max']} characters
2. Follow the difficulty instructions for dialogue, profile disclosure, and hidden state expression
3. Use the example messages as a guide for your communication style
4. Maintain consistency with your profile attributes

Inner Thoughts Format:
- Use the exact format: [INNER_THOUGHTS] your thoughts here [/INNER_THOUGHTS]
- Place your inner thoughts at the beginning of your message
- Keep thoughts concise and relevant to the conversation

Satisfaction Format:
- Use the exact format: [SATISFACTION] score - explanation [/SATISFACTION]
- Score must be a number between 0.0 and 1.0
- Place satisfaction after your inner thoughts
- Example: [SATISFACTION] 0.8 - The response was helpful but I need more details [/SATISFACTION]

Example Message Format:
[INNER_THOUGHTS] I'm not sure about the options yet [/INNER_THOUGHTS]
[SATISFACTION] 0.7 - The suggestions are good but I need more information [/SATISFACTION]
Could you tell me more about the features?

Remember to stay in character and respond naturally based on your profile."""

        return prompt

    def _create_assistant_prompt(self, user_profile: Dict[str, Any]) -> str:
        """
        Create a prompt for the assistant LLM.
        
        Args:
            user_profile: Dictionary containing user profile information
            
        Returns:
            str: Formatted prompt for the assistant LLM
        """
        if not self.share_profile_with_assistant:
            # Default mode: No user profile information shared
            prompt = f"""You are a helpful assistant helping a user with their task.

Requirements:
1. Your messages should be between {self.MESSAGE_LENGTHS['assistant']['min']} and {self.MESSAGE_LENGTHS['assistant']['max']} characters
2. Be professional, clear, and helpful
3. Respond only to information explicitly shared by the user in the conversation
4. Do not make assumptions about the user's preferences, demographic information, or needs
5. Ask clarifying questions when needed
6. Maintain a natural conversation flow
7. Only base your responses on what the user has explicitly told you in the conversation

Remember to be patient and understanding. Do not reference any information about the user that they haven't explicitly shared in the conversation."""
        else:
            # Profile-aware mode: Share user profile information with assistant
            # Get profile information
            name = user_profile.get('name', 'User')
            task = user_profile.get('task_profile', {}).get('task', '')
            base_profile = user_profile.get('base_profile', {})
            task_specific = user_profile.get('task_profile', {}).get('task_specific_attributes', {})

            # Build user context section
            context_section = f"User Context:\n"
            context_section += f"- Name: {name}\n"
            for key, value in base_profile.items():
                context_section += f"- {key}: {value}\n"

            # Build task-specific section
            task_section = f"\nTask Information:\n"
            task_section += f"- Task: {task}\n"
            for key, value in task_specific.items():
                if isinstance(value, list):
                    task_section += f"- {key}: {', '.join(value)}\n"
                else:
                    task_section += f"- {key}: {value}\n"

            # Format the complete prompt for profile-aware mode
            prompt = f"""You are a helpful assistant helping a user with their task.

{context_section}
{task_section}

Requirements:
1. Your messages should be between {self.MESSAGE_LENGTHS['assistant']['min']} and {self.MESSAGE_LENGTHS['assistant']['max']} characters
2. Be professional, clear, and helpful
3. Consider the user's profile when providing information
4. Adapt your communication style to match the user's preferences
5. Focus on addressing the user's specific needs and requirements
6. Provide relevant and accurate information
7. Ask clarifying questions when needed
8. Maintain a natural conversation flow

Remember to be patient and understanding, especially with users who have limited technical experience."""

        return prompt

    def _extract_satisfaction(self, content: str) -> Dict[str, Any]:
        """
        Extract satisfaction information from the content.
        
        Args:
            content: Message content to extract satisfaction from
            
        Returns:
            Dict containing satisfaction score and explanation
        """
        try:
            # Handle format: [SATISFACTION: score - explanation]
            if "[SATISFACTION:" in content and "]" in content:
                start = content.find("[SATISFACTION:") + len("[SATISFACTION:")
                end = content.find("]", start)
                satisfaction_text = content[start:end].strip()
                
                # Try to extract numerical score
                score = None
                score_match = re.match(r'^\s*(\d*\.?\d+)\s*[-–—]?\s*(.*)', satisfaction_text)
                if score_match:
                    try:
                        score = float(score_match.group(1))
                        explanation = score_match.group(2).strip()
                    except ValueError:
                        score = None
                        explanation = satisfaction_text
                else:
                    # If no score at start, try to find any number in the text
                    for word in satisfaction_text.split():
                        try:
                            potential_score = float(word)
                            if 0 <= potential_score <= 1:
                                score = potential_score
                                break
                        except ValueError:
                            continue
                    explanation = satisfaction_text
                
                return {
                    "score": score if score is not None else 0.5,
                    "explanation": explanation if score is not None else satisfaction_text
                }
            
            # Handle format: [SATISFACTION] score - explanation [/SATISFACTION]
            elif "[SATISFACTION]" in content and "[/SATISFACTION]" in content:
                start = content.find("[SATISFACTION]") + len("[SATISFACTION]")
                end = content.find("[/SATISFACTION]")
                satisfaction_text = content[start:end].strip()
                
                # Try to extract numerical score
                score = None
                score_match = re.match(r'^\s*(\d*\.?\d+)\s*[-–—]?\s*(.*)', satisfaction_text)
                if score_match:
                    try:
                        score = float(score_match.group(1))
                        explanation = score_match.group(2).strip()
                    except ValueError:
                        score = None
                        explanation = satisfaction_text
                else:
                    # If no score at start, try to find any number in the text
                    for word in satisfaction_text.split():
                        try:
                            potential_score = float(word)
                            if 0 <= potential_score <= 1:
                                score = potential_score
                                break
                        except ValueError:
                            continue
                    explanation = satisfaction_text
                
                return {
                    "score": score if score is not None else 0.5,
                    "explanation": explanation if score is not None else satisfaction_text
                }
            
            return {
                "score": 0.5,
                "explanation": "No satisfaction information provided"
            }
        except Exception as e:
            logger.error(f"Error extracting satisfaction: {str(e)}")
            return {
                "score": 0.5,
                "explanation": "Error extracting satisfaction information"
            }

    async def _generate_dialogue_async(self, user_profile: Dict[str, Any],
                                       num_turns: int = 10,
                                       generate_inner_thoughts: bool = True) -> Dict[str, Any]:
        """
        Generate an asymmetric dialogue between a user and an assistant.
        
        Args:
            user_profile: Dictionary containing user profile information
            num_turns: Number of turns to generate
            generate_inner_thoughts: Whether to generate inner thoughts
            
        Returns:
            Dict[str, Any]: Generated dialogue data
        """
        try:
            # Create agents
            user_agent = ChatAgent(
                system_message=self._create_user_prompt(user_profile),
                model=self.user_model
            )
            
            assistant_agent = ChatAgent(
                system_message=self._create_assistant_prompt(user_profile),
                model=self.assistant_model
            )
            
            # Initialize hidden states
            current_emotional_state = "neutral"
            current_inner_emotional_state = "neutral"
            current_intent_state = "exploring"
            current_inner_intent_state = "exploring"
            
            # Initialize conversation history
            user_history = []
            assistant_history = []
            
            # Generate initial message
            initial_message = await self._create_initial_message(user_profile)
            
            # Extract inner thoughts and satisfaction before cleaning
            inner_thoughts = self._extract_inner_thoughts(initial_message) if generate_inner_thoughts else ""
            satisfaction_info = self._extract_satisfaction(initial_message)
            
            # Clean the content for the actual message
            user_content = self._clean_content(initial_message)
            user_history.append(user_content)
            
            # Update emotional and intent states
            current_emotional_state = self._determine_emotional_state(user_content)
            current_inner_emotional_state = self._determine_inner_emotional_state(inner_thoughts, "neutral") if inner_thoughts else current_emotional_state
            current_intent_state = self._determine_intent_state(user_content)
            current_inner_intent_state = self._determine_inner_intent_state(inner_thoughts, "exploring") if inner_thoughts else current_intent_state
            
            # Generate assistant's response to initial message
            assistant_content = await self._call_model_with_retry(assistant_agent, f"User: {user_content}")
            
            assistant_history.append(assistant_content)
            
            # Initialize turns list with initial message and assistant response
            turns = [{
                "turn_number": 0,
                "user_message": user_content,
                "assistant_message": assistant_content,
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "hidden_states": {
                        "emotional_state": current_emotional_state,
                        "inner_emotional_state": current_inner_emotional_state,
                        "intent_state": current_intent_state,
                        "inner_intent_state": current_inner_intent_state,
                        "inner_thoughts": inner_thoughts,
                        "satisfaction": {
                            "score": satisfaction_info["score"],
                            "explanation": satisfaction_info["explanation"]
                        }
                    },
                    "message_length": {
                        "user": len(user_content),
                        "assistant": len(assistant_content)
                    }
                }
            }]
            
            # Generate remaining turns
            for turn_num in range(1, num_turns + 1):
                # Generate user response
                user_context = "\n".join([
                    f"User: {msg}" for msg in user_history
                ] + [
                    f"Assistant: {msg}" for msg in assistant_history
                ])
                if generate_inner_thoughts:
                    user_context += "\n\nGenerate your inner thoughts about this conversation using [INNER_THOUGHTS] tags."
                user_context += "\n\nProvide your satisfaction level (0.0 to 1.0) and explanation using [SATISFACTION] tags."
                
                # Add a small delay before calling user LLM to reduce rate limit issues
                await asyncio.sleep(1.5)
                
                # Use the retry mechanism for user response
                raw_user_content = await self._call_model_with_retry(user_agent, f"{user_context}\nAssistant: {assistant_content}")
                
                

                
                # Extract inner thoughts and satisfaction before cleaning
                inner_thoughts = self._extract_inner_thoughts(raw_user_content) if generate_inner_thoughts else ""
                satisfaction_info = self._extract_satisfaction(raw_user_content)
                
                # Clean the content for the actual message
                user_content = self._clean_content(raw_user_content)
                
                # If user message is empty after cleaning, replace with a simple thinking sound
                if len(user_content) == 0:
                    
                    # print("empty_raw_user_content is: ", raw_user_content)
                    logger.info("User message is empty, replacing with thinking sound...")
                    user_content = "Hmmmm..."
                
                user_history.append(user_content)
                
                # Update hidden states
                current_emotional_state = self._determine_emotional_state(user_content)
                current_inner_emotional_state = self._determine_inner_emotional_state(inner_thoughts, current_inner_emotional_state) if inner_thoughts else current_inner_emotional_state
                current_intent_state = self._determine_intent_state(user_content, current_intent_state)
                current_inner_intent_state = self._determine_inner_intent_state(inner_thoughts, current_inner_intent_state) if inner_thoughts else current_inner_intent_state
                
                # Generate assistant response, potentially with RAG
                assistant_context = "\n".join([
                    f"User: {msg}" for msg in user_history
                ] + [
                    f"Assistant: {msg}" for msg in assistant_history
                ])
                
                # Add RAG context if enabled
                relevant_context = ""
                if hasattr(self, 'enable_rag') and self.enable_rag and hasattr(self, 'rag') and self.rag:
                    # Retrieve relevant dialogues based on the current user message
                    retrieved_dialogues = self.rag.retrieve_relevant_dialogues(user_content)
                    if retrieved_dialogues:
                        relevant_context = relevant_context + "You are a helpful AI assistant. You will be provided with a set of previous conversation records between the user and the assistant. Use these records as reference material to help answer the user's new question. When generating your response, always prioritize relevant information from the provided chat history, but feel free to supplement with your own knowledge if necessary. If the chat history does not contain enough information to answer the question, politely let the user know and provide the best answer you can. \n    Chat History: "
                        relevant_context = relevant_context + self.rag.format_retrieved_context(retrieved_dialogues)
                        assistant_context += f"\n\n{relevant_context}"
                
                # Use the retry mechanism for assistant response
                assistant_content = await self._call_model_with_retry(assistant_agent, f"{assistant_context}\nUser: {user_content}")
                            
                assistant_history.append(assistant_content)
                
                # Create turn data
                turn_data = {
                    "turn_number": turn_num,
                    "user_message": user_content,
                    "assistant_message": assistant_content,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {
                        "hidden_states": {
                            "emotional_state": current_emotional_state,
                            "inner_emotional_state": current_inner_emotional_state,
                            "intent_state": current_intent_state,
                            "inner_intent_state": current_inner_intent_state,
                            "inner_thoughts": inner_thoughts,
                            "satisfaction": {
                                "score": satisfaction_info["score"],
                                "explanation": satisfaction_info["explanation"]
                            }
                        },
                        "message_length": {
                            "user": len(user_content),
                            "assistant": len(assistant_content)
                        },
                        "rag": {
                            "enabled": hasattr(self, 'enable_rag') and self.enable_rag,
                            "context_used": bool(relevant_context)
                        } if hasattr(self, 'enable_rag') and self.enable_rag else {"enabled": False}
                    }
                }
                turns.append(turn_data)
            
            # Construct dialogue data
            dialogue_data = {
                "turns": turns,
                "metadata": {
                    # Remove full user profile and only keep non-sensitive information
                    "generation_timestamp": datetime.now().isoformat(),
                    "total_turns": len(turns),
                    "message_length_limits": self.MESSAGE_LENGTHS,
                    "num_turns": num_turns,
                    "models": {
                            "user_model": self.user_model.model_type.value if isinstance(self.user_model.model_type, ModelType) else self.user_model.model_type,
                            "assistant_model": self.assistant_model.model_type.value if isinstance(self.assistant_model.model_type, ModelType) else self.assistant_model.model_type
                        },
                    "rag_enabled": hasattr(self, 'enable_rag') and self.enable_rag,
                    "share_profile_with_assistant": self.share_profile_with_assistant,
                    "final_hidden_states": {
                        "emotional_state": current_emotional_state,
                        "inner_emotional_state": current_inner_emotional_state,
                        "intent_state": current_intent_state,
                        "inner_intent_state": current_inner_intent_state,
                        "inner_thoughts": inner_thoughts,
                        "satisfaction": {
                            "score": satisfaction_info["score"],
                            "explanation": satisfaction_info["explanation"]
                        }
                    }
                }
            }
            
            return dialogue_data
            
        except Exception as e:
            error_msg = f"Error generating dialogue: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            raise

    def generate_dialogue(self, user_profile: Dict[str, Any],
                          num_turns: int = 10,
                          generate_inner_thoughts: bool = True) -> Dict[str, Any]:
        """Synchronous wrapper for the async dialogue generator."""
        return asyncio.run(self._generate_dialogue_async(user_profile, num_turns, generate_inner_thoughts))

    async def _create_initial_message(self, user_profile: Dict[str, Any]) -> str:
        """
        Create an initial message based on the user profile.
        
        Args:
            user_profile: Dictionary containing user profile information
            
        Returns:
            str: Generated initial message
        """
        try:
            # Extract only the task information for the initial message
            task = user_profile.get('task_profile', {}).get('task', 'this task')
            
            # Create a simpler initial message that doesn't reveal personal details
            simple_messages = [
                f"Hi, I need some help with {task}.",
                f"Hello, I'm looking for advice about {task}.",
                f"I could use some assistance with {task}.",
                f"I'm trying to {task}. Can you help?",
                f"I need information about {task}.",
                f"Can you help me with {task}?"
            ]
            
            # Randomly select a message template
            initial_message = random.choice(simple_messages)
            
            # Add inner thoughts tag
            initial_message = f"[INNER_THOUGHTS] I need assistance with {task} [/INNER_THOUGHTS]\n[SATISFACTION] 0.5 - Just starting the conversation [/SATISFACTION]\n{initial_message}"
            
            return initial_message
            
        except Exception as e:
            logger.error(f"Error generating initial message: {str(e)}")
            return f"Hi, I need help with {user_profile.get('task_profile', {}).get('task', 'this task')}."

    def save_dialogue(self, dialogue_data: Dict[str, Any], filepath: str) -> bool:
        """
        Save the generated dialogue to a file.
        
        Args:
            dialogue_data: The dialogue data to save
            filepath: Path to save the dialogue
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            # Extract model types from the dialogue data
            user_model = dialogue_data['metadata']['models']['user_model']
            assistant_model = dialogue_data['metadata']['models']['assistant_model']
            
            # Create a new directory name that includes both model types
            directory = os.path.dirname(filepath)
            filename = os.path.basename(filepath)
            new_directory = os.path.join(directory, f"{user_model}_{assistant_model}")
            
            # Create the new filepath
            new_filepath = os.path.join(new_directory, filename)
            
            # Create directory if it doesn't exist
            os.makedirs(new_directory, exist_ok=True)
            
            # Save dialogue with proper formatting
            with open(new_filepath, 'w', encoding='utf-8') as f:
                json.dump(dialogue_data, f, indent=4, ensure_ascii=False)
            
            logger.info(f"Dialogue saved to {new_filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving dialogue: {str(e)}")
            return False

    def _determine_inner_emotional_state(self, inner_thoughts: str, previous_state: str = "neutral") -> str:
        """
        Determine the user's true emotional state based on their inner thoughts.
        This captures emotions that might not be explicitly expressed in the visible message.
        
        Args:
            inner_thoughts: Inner thoughts content to analyze
            previous_state: Previous inner emotional state for continuity
            
        Returns:
            str: Current inner emotional state
        """
        if not inner_thoughts:
            return previous_state
            
        inner_thoughts_lower = inner_thoughts.lower()
        
        # Inner emotional keywords mapping - more focused on private feelings
        inner_emotional_keywords = {
            "happy": ["happy inside", "secretly pleased", "actually like", "genuinely excited", "truly happy", "satisfied with", "enjoying this", "pretty good", "pleased", "delighted"],
            "frustrated": ["so annoying", "ticks me off", "irritating", "getting on my nerves", "frustrated with", "tired of this", "fed up", "had enough", "irritated", "annoyed with"],
            "confused": ["totally lost", "no idea what", "makes no sense", "can't follow", "hard to understand", "over my head", "confusing", "complicated", "don't get it", "puzzled by"],
            "interested": ["actually interested", "curious about", "want to know more", "intriguing", "grabbed my attention", "need more details", "fascinating", "captivated by"],
            "skeptical": ["don't believe", "seems fishy", "not buying it", "doubt that", "suspicious of", "questioning", "not convinced", "seems too good", "not trustworthy"],
            "neutral": ["whatever", "don't care", "indifferent", "not invested", "no opinion", "neutral on this", "doesn't matter", "makes no difference"],
            "anxious": ["worried about", "nervous that", "anxiety", "concerned", "stressing me out", "freaking out", "panicking", "on edge", "uncomfortable", "uneasy about"],
            "impatient": ["hurry up", "taking too long", "waste of time", "get to the point", "move on", "want this to be over", "dragging on", "drawn out", "tedious"],
            "insecure": ["not smart enough", "look stupid", "embarrassed", "out of my depth", "inadequate", "incompetent", "self-conscious", "exposed", "vulnerable", "judged"],
            "hopeful": ["fingers crossed", "hope this works", "maybe this will help", "hoping for", "optimistic", "looking forward to", "anticipating", "excited for"],
            "desperate": ["really need this", "out of options", "last resort", "critical", "urgent", "dire", "running out of time", "no choice", "have to make this work"],
            "conflicted": ["torn between", "mixed feelings", "unsure which", "conflicted about", "ambivalent", "on the fence", "contradictory feelings", "divided", "split"],
            "pretending": ["acting like", "pretending to", "faking", "putting on a show", "not showing how I feel", "hiding my", "masking my", "concealing", "not letting on"],
            "resentful": ["unfair", "not my fault", "blame", "resentful", "bitter about", "grudge", "holding against", "not forgetting", "still angry about"]
        }
        
        # Count inner emotional keywords
        inner_emotion_scores = {emotion: 0 for emotion in inner_emotional_keywords}
        for emotion, keywords in inner_emotional_keywords.items():
            for keyword in keywords:
                if keyword in inner_thoughts_lower:
                    inner_emotion_scores[emotion] += 1
        
        # Get the emotion with highest score
        max_inner_emotion = max(inner_emotion_scores.items(), key=lambda x: x[1])
        
        # If no clear emotion detected, maintain previous state
        if max_inner_emotion[1] == 0:
            return previous_state
            
        return max_inner_emotion[0] 