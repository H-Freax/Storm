import os
import sys
from typing import Dict, Any, List, Optional
import asyncio
from dotenv import load_dotenv
import json
import logging
from datetime import datetime
import traceback
from camel.agents import ChatAgent
from camel.types import ModelType
from camel.models import ModelFactory
from camel.configs import ChatGPTConfig
from .task_profile_generator import TaskProfileGenerator
from .random_profile_generator import RandomProfileGenerator
import random

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

class UserProfileGenerator:
    """
    A class for generating user profiles based on task and difficulty level.
    This class uses LLM to generate realistic and consistent user profiles.
    """

    def __init__(self, model_type: ModelType = ModelType.GPT_4O_MINI):
        """
        Initialize the user profile generator.
        
        Args:
            model_type: The type of model to use for generation
        """
        try:
            # Get OpenAI API key from environment variables
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")

            # Configure model settings
            model_config = ChatGPTConfig(
                temperature=1,  # Balanced temperature for creative but consistent profiles
                max_tokens=2000,  # Sufficient tokens for profile generation
                presence_penalty=0.1,  # Slight penalty for repeating tokens
                frequency_penalty=0.1  # Slight penalty for frequent token usage
            )

            # Create model instance
            self.model = ModelFactory.create(
                model_platform="openai",
                model_type=model_type,
                model_config_dict=model_config.as_dict(),
                api_key=api_key,
                timeout=60.0  # 1-minute timeout for API calls
            )
            
            # Create task profile generator
            self.task_profile_generator = TaskProfileGenerator(model_type=model_type)
            
            # Create random profile generator
            self.random_profile_generator = RandomProfileGenerator()
            
            logger.info("Successfully initialized user profile generator")
            
        except Exception as e:
            error_msg = f"Error initializing user profile generator: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            raise

    def _extract_message_content(self, response) -> str:
        """
        Extract content from various response types.
        
        Args:
            response: The response object from the model
            
        Returns:
            str: The extracted content
        """
        try:
            # Get the raw content
            if hasattr(response, 'msg') and hasattr(response.msg, 'content'):
                content = response.msg.content
            elif hasattr(response, 'content'):
                content = response.content
            elif hasattr(response, 'msgs') and response.msgs:
                content = response.msgs[-1].content
            else:
                content = str(response)

            # Early check for empty content
            if not content or not content.strip():
                logger.warning("Empty content received from model")
                return ""
                
            return content
        except Exception as e:
            logger.error(f"Error extracting message content: {str(e)}")
            return ""

    def _validate_user_profile(self, profile: Dict[str, Any]) -> bool:
        """
        Validate the user profile structure.
        
        Args:
            profile: The user profile to validate
            
        Returns:
            bool: True if profile is valid, False otherwise
        """
        try:
            required_fields = {
                'name': str,
                'description': str,
                'base_profile': dict,
                'behavioral_traits': dict,
                'contextual_factors': dict,
                'task_profile': dict,
                'example_dimensions': dict  # Add example dimensions field
            }
            
            # Check top-level fields
            for field, field_type in required_fields.items():
                if field not in profile or not isinstance(profile[field], field_type):
                    logger.error(f"Missing or invalid field: {field}")
                    return False
            
            # Check base profile fields
            base_profile = profile['base_profile']
            required_base_fields = {
                'age_group': str,
                'tech_experience': str,
                'language_style': str,
                'personality': str,
                'culture': str,
                'decision_style': str,
                'communication_style': str,
                'expressiveness': str,
                'social_context': str,
                'physical_status': str
            }
            
            for field, field_type in required_base_fields.items():
                if field not in base_profile or not isinstance(base_profile[field], field_type):
                    logger.error(f"Missing or invalid base profile field: {field}")
                    return False
            
            # Check behavioral traits
            behavioral_traits = profile['behavioral_traits']
            required_trait_fields = {
                'patience': str,
                'attention_to_detail': str,
                'risk_tolerance': str,
                'adaptability': str,
                'learning_style': str
            }
            
            for field, field_type in required_trait_fields.items():
                if field not in behavioral_traits or not isinstance(behavioral_traits[field], field_type):
                    logger.error(f"Missing or invalid behavioral trait field: {field}")
                    return False
            
            # Check contextual factors
            contextual_factors = profile['contextual_factors']
            required_context_fields = {
                'time_constraint': str,
                'environment': str,
                'social_pressure': str,
                'previous_experience': str
            }
            
            for field, field_type in required_context_fields.items():
                if field not in contextual_factors or not isinstance(contextual_factors[field], field_type):
                    logger.error(f"Missing or invalid contextual factor field: {field}")
                    return False

            # Check example dimensions
            example_dimensions = profile['example_dimensions']
            required_example_fields = {
                'style': str,
                'length': str,
                'content': str,
                'tone': str,
                'examples': list
            }
            
            for field, field_type in required_example_fields.items():
                if field not in example_dimensions or not isinstance(example_dimensions[field], field_type):
                    logger.error(f"Missing or invalid example dimension field: {field}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating user profile: {str(e)}")
            return False

    async def generate_profile(self, task: str, difficulty_level: int = 1) -> Dict[str, Any]:
        """
        Generate a user profile based on task and difficulty level.
        
        Args:
            task: The task description
            difficulty_level: The difficulty level for profile generation
            
        Returns:
            Dict containing the generated profile
        """
        try:
            # Generate random profile combination
            profile = await self.generate_random_profile_combination(task, difficulty_level)
            
            # Create a prompt for generating name and description
            generation_prompt = f"""Based on the following user profile, generate a realistic name and description:

Base Profile:
{json.dumps(profile['base_profile'], indent=2)}

Behavioral Traits:
{json.dumps(profile['behavioral_traits'], indent=2)}

Contextual Factors:
{json.dumps(profile['contextual_factors'], indent=2)}

Task: {task}
Difficulty Level: {difficulty_level}

Generate a response in the following JSON format:
{{
    "name": "Realistic name that matches the profile",
    "description": "A detailed description of the user's background, personality, and current situation"
}}

Requirements:
1. The name should be culturally appropriate based on the profile
2. The description should be detailed and consistent with all profile attributes
3. The description should explain why they are interested in the task
4. Keep the description concise but informative (2-3 sentences)
"""

            # Create a temporary ChatAgent for generating name and description
            temp_agent = ChatAgent(
                system_message="""You are a profile generator for dialogue systems.
Your task is to generate realistic names and descriptions based on user profiles.
Focus on creating consistent and culturally appropriate profiles.
Always respond with valid JSON containing name and description.""",
                model=self.model
            )
            
            # Get response from LLM
            response = temp_agent.step(generation_prompt)
            content = self._extract_message_content(response)
            
            # Clean the content to ensure it's valid JSON
            content = content.strip()
            # Handle markdown code blocks (both with and without language specification)
            if content.startswith('```json'):
                content = content[7:]
            elif content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            # Additional cleaning for common issues
            # Remove any text before the first { or after the last }
            first_brace = content.find('{')
            last_brace = content.rfind('}')
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                content = content[first_brace:last_brace+1]
            
            # Parse the JSON response
            try:
                if not content:
                    raise ValueError("Empty content received from model")
                    
                result = json.loads(content)
                if not isinstance(result, dict) or 'name' not in result or 'description' not in result:
                    raise ValueError("Invalid JSON structure")
                
                # Update profile with generated name and description
                profile['name'] = result['name']
                profile['description'] = result['description']
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON response: {str(e)}")
                logger.error(f"Raw content: {content}")
                # Use default values if parsing fails
                profile['name'] = "User"
                profile['description'] = f"A user interested in {task}"
            
            # Validate the profile
            if not self._validate_user_profile(profile):
                raise ValueError("Invalid user profile generated")
            
            return profile
            
        except Exception as e:
            logger.error(f"Error generating user profile: {str(e)}")
            raise

    def save_profile(self, profile: Dict[str, Any], filepath: str) -> bool:
        """
        Save the generated profile to a file.
        
        Args:
            profile: The profile to save
            filepath: Path to save the profile
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            # Validate profile before saving
            if not self._validate_user_profile(profile):
                logger.error("Invalid user profile, not saving")
                return False
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save profile with proper formatting
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(profile, f, indent=4, ensure_ascii=False)
            
            logger.info(f"Profile saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving profile: {str(e)}")
            return False

    def load_profile(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        Load a profile from a file.
        
        Args:
            filepath: Path to the profile file
            
        Returns:
            Optional[Dict[str, Any]]: Loaded profile if successful, None otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(filepath):
                logger.error(f"Profile file not found: {filepath}")
                return None
            
            # Load profile
            with open(filepath, 'r', encoding='utf-8') as f:
                profile = json.load(f)
            
            # Validate loaded profile
            if not self._validate_user_profile(profile):
                logger.error("Invalid user profile loaded")
                return None
            
            logger.info(f"Profile loaded from {filepath}")
            return profile
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing profile file: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error loading profile: {str(e)}")
            return None

    def get_all_user_profiles(self, directory: str) -> List[Dict[str, Any]]:
        """
        Get all user profiles from a directory.
        
        Args:
            directory: Directory containing user profile files
            
        Returns:
            List of user profiles
        """
        try:
            profiles = []
            if not os.path.exists(directory):
                logger.error(f"Directory not found: {directory}")
                return profiles
            
            # Get all JSON files in the directory
            for filename in os.listdir(directory):
                if filename.endswith('.json'):
                    filepath = os.path.join(directory, filename)
                    profile = self.load_profile(filepath)
                    if profile:
                        profiles.append(profile)
            
            logger.info(f"Loaded {len(profiles)} user profiles from {directory}")
            return profiles
            
        except Exception as e:
            logger.error(f"Error getting user profiles: {str(e)}")
            return []

    def get_random_user_profile(self, directory: str) -> Optional[Dict[str, Any]]:
        """
        Get a random user profile from a directory.
        
        Args:
            directory: Directory containing user profile files
            
        Returns:
            Optional[Dict[str, Any]]: Random user profile if successful, None otherwise
        """
        try:
            profiles = self.get_all_user_profiles(directory)
            if not profiles:
                return None
            
            return random.choice(profiles)
            
        except Exception as e:
            logger.error(f"Error getting random user profile: {str(e)}")
            return None

    def get_random_task_profile(self, directory: str) -> Optional[Dict[str, Any]]:
        """
        Get a random task profile from a directory.
        
        Args:
            directory: Directory containing task profile files
            
        Returns:
            Optional[Dict[str, Any]]: Random task profile if successful, None otherwise
        """
        try:
            profiles = self.task_profile_generator.get_all_task_profiles(directory)
            if not profiles:
                return None
            
            return random.choice(profiles)
            
        except Exception as e:
            logger.error(f"Error getting random task profile: {str(e)}")
            return None

    async def generate_random_profile_combination(self, task: str, difficulty_level: int = 1) -> Dict[str, Any]:
        """
        Generate a random combination of user profiles.
        
        Args:
            task: The task description
            difficulty_level: The difficulty level for profile generation
            
        Returns:
            Dict containing the combined profile
        """
        try:
            # Get random base profile
            base_profile = self.random_profile_generator.generate_profile()
            
            # Get example dimensions based on difficulty level
            example_dimensions = self.random_profile_generator.get_example_dimensions(difficulty_level)
            
            # Get difficulty instructions
            difficulty_instructions = self.random_profile_generator.get_difficulty_instructions(difficulty_level)
            
            # Generate task-specific attributes using GPT
            task_prompt = f"""Based on the following task and user profile, generate task-specific attributes:

Task: {task}
Base Profile:
{json.dumps(base_profile, indent=2)}

Generate a response in the following JSON format:
{{
    "task_specific_attributes": {{
        "budget_range": "string",
        "priority_features": ["string"],
        "usage_scenarios": ["string"],
        "preferred_brands": ["string"],
        "timeline": "string",
        "purchase_location": "string",
        "additional_requirements": ["string"]
    }}
}}

Requirements:
1. Attributes should be specific to the task and consistent with the user profile
2. Consider the user's tech experience, personality, and behavioral traits
3. Make the attributes realistic and detailed
4. Include at least 3 priority features and usage scenarios
5. IMPORTANT: Your response must be valid JSON only, with no additional text or explanation
"""

            # Create a temporary ChatAgent for generating task-specific attributes
            temp_agent = ChatAgent(
                system_message="""You are a task-specific profile generator.
Your task is to generate realistic task-specific attributes based on user profiles and tasks.
Focus on creating consistent and detailed attributes that match the user's profile.
IMPORTANT: You must respond with ONLY valid JSON containing task-specific attributes, with no additional text or explanation.""",
                model=self.model
            )
            
            # Get response from LLM
            response = temp_agent.step(task_prompt)
            content = self._extract_message_content(response)
            
            # Clean the content to ensure it's valid JSON
            content = content.strip()
            # Handle markdown code blocks (both with and without language specification)
            if content.startswith('```json'):
                content = content[7:]
            elif content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            # Additional cleaning for common issues
            # Remove any text before the first { or after the last }
            first_brace = content.find('{')
            last_brace = content.rfind('}')
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                content = content[first_brace:last_brace+1]
            
            # Parse the JSON response
            try:
                if not content:
                    raise ValueError("Empty content received from model")
                    
                result = json.loads(content)
                if not isinstance(result, dict) or 'task_specific_attributes' not in result:
                    raise ValueError("Invalid JSON structure")
                task_specific_attributes = result['task_specific_attributes']
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Error parsing task-specific attributes: {str(e)}")
                logger.error(f"Raw content: {content}")
                # Use fallback default attributes
                task_specific_attributes = {
                    "budget_range": "Not specified",
                    "priority_features": ["Basic functionality"],
                    "usage_scenarios": ["General use"],
                    "preferred_brands": ["Any"],
                    "timeline": "Not specified",
                    "purchase_location": "Not specified",
                    "additional_requirements": ["None"]
                }
            
            # Combine all profiles
            combined_profile = {
                "name": base_profile.get("name", "User"),
                "description": base_profile.get("description", ""),
                "base_profile": {
                    "age_group": base_profile.get("base_profile", {}).get("age_group", ""),
                    "tech_experience": base_profile.get("base_profile", {}).get("tech_experience", ""),
                    "language_style": base_profile.get("base_profile", {}).get("language_style", ""),
                    "personality": base_profile.get("base_profile", {}).get("personality", ""),
                    "culture": base_profile.get("base_profile", {}).get("culture", ""),
                    "decision_style": base_profile.get("base_profile", {}).get("decision_style", ""),
                    "communication_style": base_profile.get("base_profile", {}).get("communication_style", ""),
                    "expressiveness": base_profile.get("base_profile", {}).get("expressiveness", ""),
                    "social_context": base_profile.get("base_profile", {}).get("social_context", ""),
                    "physical_status": base_profile.get("base_profile", {}).get("physical_status", "")
                },
                "behavioral_traits": base_profile.get("behavioral_traits", {}),
                "contextual_factors": base_profile.get("contextual_factors", {}),
                "task_profile": {
                    "task": task,
                    "difficulty_level": difficulty_level,
                    "instructions": difficulty_instructions,
                    "task_specific_attributes": task_specific_attributes
                },
                "example_dimensions": example_dimensions
            }
            
            # Add metadata
            combined_profile["metadata"] = {
                "generation_timestamp": datetime.now().isoformat(),
                "task": task,
                "difficulty_level": difficulty_level,
                "model_type": self.model.model_type.name
            }
            
            return combined_profile
            
        except Exception as e:
            logger.error(f"Error generating combined profile: {str(e)}")
            raise 