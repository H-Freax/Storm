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

class TaskProfileGenerator:
    """
    A class for generating task-specific profiles based on task type and difficulty level.
    This class uses LLM to generate realistic task-specific attributes and requirements.
    """

    # Predefined option pools for different aspects of task profiles
    PREFERENCE_POOLS = {
        "must_have": [
            "High quality and durability",
            "Latest technology and features",
            "Good value for money",
            "Brand reputation",
            "Ease of use",
            "Compatibility with existing devices",
            "Long battery life",
            "Fast performance",
            "Good customer support",
            "Warranty coverage",
            "Environmentally friendly",
            "Customization options",
            "Future-proof design",
            "Security features",
            "User-friendly interface",
            "Portability",
            "Reliability",
            "Energy efficiency",
            "Maintenance requirements",
            "Upgradeability"
        ],
        "nice_to_have": [
            "Premium design",
            "Advanced features",
            "Smart home integration",
            "Cloud storage",
            "Wireless charging",
            "Water resistance",
            "Fingerprint sensor",
            "Face recognition",
            "AI capabilities",
            "Virtual assistant",
            "Gaming features",
            "Professional tools",
            "Creative software",
            "Collaboration features",
            "Remote access",
            "Backup solutions",
            "Multi-device sync",
            "Custom themes",
            "Accessibility features",
            "Health monitoring"
        ],
        "deal_breakers": [
            "Poor quality",
            "High maintenance",
            "Limited warranty",
            "Poor customer service",
            "Compatibility issues",
            "Security concerns",
            "Short lifespan",
            "Difficult to use",
            "Expensive repairs",
            "Limited support",
            "Poor performance",
            "Battery issues",
            "Overheating problems",
            "Software bugs",
            "Privacy concerns",
            "Limited storage",
            "Slow updates",
            "Restrictive policies",
            "Poor connectivity",
            "Limited customization"
        ]
    }

    BUDGET_FLEXIBILITY = [
        "Very flexible - willing to pay more for better quality",
        "Somewhat flexible - can adjust for important features",
        "Moderate - prefer to stay within range but can be convinced",
        "Limited - strict budget constraints",
        "Fixed - cannot exceed budget under any circumstances",
        "Open-ended - quality is more important than cost",
        "Value-focused - looking for best price-performance ratio",
        "Premium - willing to pay for top-tier options",
        "Budget-conscious - seeking best deals",
        "Investment-minded - considering long-term value"
    ]

    PAYMENT_METHODS = [
        "Credit card",
        "Debit card",
        "Bank transfer",
        "PayPal",
        "Digital wallet",
        "Cash",
        "Installment plan",
        "Lease option",
        "Trade-in",
        "Gift cards",
        "Cryptocurrency",
        "Company account",
        "Financing",
        "Layaway",
        "Subscription"
    ]

    KNOWLEDGE_LEVELS = [
        "Expert - very knowledgeable in the field",
        "Advanced - good understanding of technical aspects",
        "Intermediate - familiar with basic concepts",
        "Beginner - limited knowledge but eager to learn",
        "Novice - completely new to the subject",
        "Professional - industry experience",
        "Enthusiast - self-taught with practical experience",
        "Student - learning and researching",
        "Casual user - basic understanding",
        "Uncertain - not sure about technical details"
    ]

    URGENCY_LEVELS = [
        "Immediate - needed right away",
        "Urgent - within a few days",
        "Soon - within a week",
        "Planned - within a month",
        "Future - planning ahead",
        "Flexible - no strict timeline",
        "Research phase - gathering information",
        "Comparison phase - evaluating options",
        "Decision phase - ready to choose",
        "Exploratory - just starting to look"
    ]

    DECISION_FACTORS = [
        "Price and budget",
        "Quality and durability",
        "Features and functionality",
        "Brand reputation",
        "User reviews",
        "Technical specifications",
        "Design and aesthetics",
        "Ease of use",
        "Customer support",
        "Warranty and protection",
        "Future compatibility",
        "Environmental impact",
        "Social proof",
        "Personal preferences",
        "Professional requirements",
        "Lifestyle fit",
        "Long-term value",
        "Maintenance needs",
        "Security features",
        "Innovation level"
    ]

    def __init__(self, model_type: ModelType = ModelType.GPT_4O_MINI):
        """
        Initialize the task profile generator.
        
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
                temperature=0.7,  # Balanced temperature for creative but consistent profiles
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
            
            logger.info("Successfully initialized task profile generator")
            
        except Exception as e:
            error_msg = f"Error initializing task profile generator: {str(e)}\n{traceback.format_exc()}"
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

            # Remove markdown code block markers if present
            if content.startswith('```'):
                # Remove the first line (```json or similar)
                content = '\n'.join(content.split('\n')[1:])
            if content.endswith('```'):
                # Remove the last line (```)
                content = '\n'.join(content.split('\n')[:-1])

            return content.strip()
            
        except Exception as e:
            logger.error(f"Error extracting message content: {str(e)}")
            return str(response)

    def _validate_task_profile(self, profile: Dict[str, Any]) -> bool:
        """
        Validate the task profile structure.
        
        Args:
            profile: The task profile to validate
            
        Returns:
            bool: True if profile is valid, False otherwise
        """
        try:
            required_fields = {
                'task_name': str,
                'task_description': str,
                'budget': dict,
                'preferences': dict,
                'constraints': dict,
                'knowledge_level': str,
                'urgency': str,
                'decision_factors': list,
                'task_requirements': dict,
                'success_criteria': dict
            }
            
            # Check top-level fields
            for field, field_type in required_fields.items():
                if field not in profile or not isinstance(profile[field], field_type):
                    logger.error(f"Missing or invalid field: {field}")
                    return False
            
            # Check budget fields
            budget = profile['budget']
            required_budget_fields = {
                'range': dict,
                'flexibility': str,
                'payment_methods': list
            }
            for field, field_type in required_budget_fields.items():
                if field not in budget or not isinstance(budget[field], field_type):
                    logger.error(f"Missing or invalid budget field: {field}")
                    return False
            
            # Check preferences fields
            preferences = profile['preferences']
            required_preference_fields = {
                'must_have': list,
                'nice_to_have': list,
                'deal_breakers': list
            }
            for field, field_type in required_preference_fields.items():
                if field not in preferences or not isinstance(preferences[field], field_type):
                    logger.error(f"Missing or invalid preference field: {field}")
                    return False
            
            # Check success criteria fields
            success_criteria = profile['success_criteria']
            required_criteria_fields = {
                'must_meet': list,
                'should_meet': list,
                'nice_to_meet': list
            }
            for field, field_type in required_criteria_fields.items():
                if field not in success_criteria or not isinstance(success_criteria[field], field_type):
                    logger.error(f"Missing or invalid success criteria field: {field}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating task profile: {str(e)}")
            return False

    async def _generate_options_pool(self, task: str, option_type: str) -> List[str]:
        """
        Generate a pool of options for a specific type using GPT.
        
        Args:
            task: The task description
            option_type: Type of options to generate (e.g., "must_have", "nice_to_have", etc.)
            
        Returns:
            List of generated options
        """
        prompt = f"""Generate a diverse list of {option_type} options for the task: {task}

Requirements:
1. Generate 15-20 unique and realistic options
2. Include both common and unique scenarios
3. Consider different user perspectives and needs
4. Make options specific to the task context
5. Include some complex and challenging options
6. Add one "Unknown/Not sure" option at the end

Format: Return a JSON array of strings.
Example: ["Option 1", "Option 2", "Unknown/Not sure"]

Write ONLY the JSON array. Do not include any explanations."""

        try:
            agent = ChatAgent(
                system_message="""You are an option generator for dialogue systems.
Your task is to generate diverse and realistic options for different aspects of tasks.
Always respond with valid JSON arrays containing the generated options.""",
                model=self.model
            )
            
            response = agent.step(prompt)
            content = self._extract_message_content(response)
            
            try:
                options = json.loads(content)
                if not isinstance(options, list):
                    raise ValueError("Response is not a list")
                return options
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing options JSON: {str(e)}")
                return ["Unknown/Not sure"]
                
        except Exception as e:
            logger.error(f"Error generating options: {str(e)}")
            return ["Unknown/Not sure"]

    async def _generate_random_preferences(self, task: str, difficulty_level: int) -> Dict[str, List[str]]:
        """Generate random preferences with increased ambiguity based on difficulty level."""
        # Generate option pools for each preference type
        must_have_pool = await self._generate_options_pool(task, "must-have preferences")
        nice_to_have_pool = await self._generate_options_pool(task, "nice-to-have preferences")
        deal_breakers_pool = await self._generate_options_pool(task, "deal-breakers")
        
        # Add vague options
        vague_options = [
            "Something that just works well",
            "Good overall quality",
            "Not too expensive but decent",
            "I heard good things about...",
            "Nothing too complicated",
            "Whatever most people recommend",
            "I'm not really sure what's important",
            "Something modern looking"
        ]
        
        # Add contradictory options based on difficulty
        contradictory_pairs = [
            ["Affordable price", "Premium quality"],
            ["Simple to use", "Advanced features"],
            ["Compact size", "Large screen/capacity"],
            ["Latest technology", "Proven reliability"]
        ]
        
        # Calculate how many vague/unknown items to include based on difficulty
        vague_count = min(difficulty_level, 3)
        unknown_count = difficulty_level // 2
        
        # Adjust selections based on difficulty level
        preferences = {
            "must_have": random.sample(must_have_pool, min(3, len(must_have_pool))),
            "nice_to_have": random.sample(nice_to_have_pool, min(4, len(nice_to_have_pool))),
            "deal_breakers": random.sample(deal_breakers_pool, min(2, len(deal_breakers_pool)))
        }
        
        # Add vague options
        if difficulty_level > 2:
            preferences["must_have"] = random.sample(vague_options, vague_count) + preferences["must_have"][:3-vague_count]
            
        # Add "Unknown/Not sure" options
        if difficulty_level > 3:
            preferences["must_have"] = ["Unknown/Not sure"] + preferences["must_have"][:2]
            
        # Add contradictory requirements at high difficulty
        if difficulty_level > 4:
            contradiction = random.choice(contradictory_pairs)
            if random.choice([True, False]):
                preferences["must_have"] = [contradiction[0]] + preferences["must_have"][:2]
                preferences["nice_to_have"] = [contradiction[1]] + preferences["nice_to_have"][:3]
            else:
                preferences["must_have"] = [contradiction[1]] + preferences["must_have"][:2]
                preferences["nice_to_have"] = [contradiction[0]] + preferences["nice_to_have"][:3]
                
        return preferences

    async def _original_generate_budget(self, task: str) -> Dict[str, Any]:
        """
        Generate random budget information using GPT.
        
        Args:
            task: The task description
            
        Returns:
            Dict containing budget information
        """
        prompt = f"""Generate budget information for the task: {task}

Requirements:
1. Generate a JSON object with the following structure:
{{
    "range": {{
        "min": number,
        "max": number
    }},
    "flexibility": "string",
    "payment_methods": ["string"]
}}

2. Consider:
   - Realistic price ranges for the task
   - Different budget flexibility levels
   - Various payment methods
   - Include "Unknown/Not sure" as a possible flexibility option

Write ONLY the JSON response. Do not include any explanations."""

        try:
            agent = ChatAgent(
                system_message="""You are a budget generator for dialogue systems.
Your task is to generate realistic budget information for different tasks.
Always respond with valid JSON that matches the required structure.""",
                model=self.model
            )
            
            response = agent.step(prompt)
            content = self._extract_message_content(response)
            
            try:
                budget_info = json.loads(content)
                # Ensure "Unknown/Not sure" is included in flexibility options
                if "Unknown/Not sure" not in budget_info.get("flexibility", ""):
                    budget_info["flexibility"] = random.choice([
                        budget_info["flexibility"],
                        "Unknown/Not sure"
                    ])
                return budget_info
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing budget JSON: {str(e)}")
                return {
                    "range": {"min": 0, "max": 0},
                    "flexibility": "Unknown/Not sure",
                    "payment_methods": ["Unknown/Not sure"]
                }
                
        except Exception as e:
            logger.error(f"Error generating budget: {str(e)}")
            return {
                "range": {"min": 0, "max": 0},
                "flexibility": "Unknown/Not sure",
                "payment_methods": ["Unknown/Not sure"]
            }

    async def _generate_random_budget(self, task: str, difficulty_level: int) -> Dict[str, Any]:
        """Generate budget with increasing uncertainty at higher difficulty levels."""
        # Original code to generate budget
        budget_info = await self._original_generate_budget(task)
        
        # Modify budget based on difficulty level
        if difficulty_level >= 3:
            # Make range wider/vaguer at higher difficulties
            if "range" in budget_info:
                min_val = budget_info["range"]["min"]
                max_val = budget_info["range"]["max"]
                # Widen the range by 30% per difficulty level above 2
                widening_factor = 1 + 0.3 * (difficulty_level - 2)
                budget_info["range"]["min"] = int(min_val / widening_factor)
                budget_info["range"]["max"] = int(max_val * widening_factor)
                
        # At difficulty 4-5, introduce complete uncertainty
        if difficulty_level >= 4 and random.random() < 0.7:
            budget_info["flexibility"] = "Unknown/Not sure"
            
        if difficulty_level == 5 and random.random() < 0.5:
            budget_info["range"] = {"min": 0, "max": 0, "note": "Completely unsure about budget"}
            
        return budget_info

    async def _generate_random_metadata(self, task: str, difficulty_level: int) -> Dict[str, Any]:
        """
        Generate random metadata using GPT.
        
        Args:
            task: The task description
            difficulty_level: The difficulty level
            
        Returns:
            Dict containing metadata
        """
        prompt = f"""Generate metadata for the task: {task} with difficulty level: {difficulty_level}

Requirements:
1. Generate a JSON object with the following structure:
{{
    "knowledge_level": "string",
    "urgency": "string",
    "decision_factors": ["string"]
}}

2. Consider:
   - Different knowledge levels (include "Unknown/Not sure")
   - Various urgency levels (include "Exploratory/Not sure")
   - Multiple decision factors
   - Task-specific considerations
   - Difficulty level implications

Write ONLY the JSON response. Do not include any explanations."""

        try:
            agent = ChatAgent(
                system_message="""You are a metadata generator for dialogue systems.
Your task is to generate realistic metadata for different tasks.
Always respond with valid JSON that matches the required structure.""",
                model=self.model
            )
            
            response = agent.step(prompt)
            content = self._extract_message_content(response)
            
            try:
                metadata = json.loads(content)
                # Ensure "Unknown" options are included
                if "Unknown/Not sure" not in metadata.get("knowledge_level", ""):
                    metadata["knowledge_level"] = random.choice([
                        metadata["knowledge_level"],
                        "Unknown/Not sure"
                    ])
                if "Exploratory/Not sure" not in metadata.get("urgency", ""):
                    metadata["urgency"] = random.choice([
                        metadata["urgency"],
                        "Exploratory/Not sure"
                    ])
                return metadata
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing metadata JSON: {str(e)}")
                return {
                    "knowledge_level": "Unknown/Not sure",
                    "urgency": "Exploratory/Not sure",
                    "decision_factors": ["Unknown/Not sure"]
                }
                
        except Exception as e:
            logger.error(f"Error generating metadata: {str(e)}")
            return {
                "knowledge_level": "Unknown/Not sure",
                "urgency": "Exploratory/Not sure",
                "decision_factors": ["Unknown/Not sure"]
            }

    async def generate_task_options(self, task: str, num_options: int = 5, 
                                  difficulty_variation: bool = True,
                                  include_uncertain: bool = True) -> List[Dict[str, Any]]:
        """Modified to increase overall profile ambiguity."""
        try:
            task_profiles = []
            for i in range(num_options):
                try:
                    # Increase base difficulty level
                    if difficulty_variation:
                        difficulty_level = (i % 5) + 1
                    else:
                        difficulty_level = 3
                    
                    # Pass difficulty level to all generator methods
                    base_profile = {
                        "task_name": task,
                        "task_description": f"Task to {task}",
                        "budget": await self._generate_random_budget(task, difficulty_level),
                        "preferences": await self._generate_random_preferences(task, difficulty_level),
                        "constraints": {
                            "time": f"Time constraint level {random.randint(1, 5)}",
                            "location": "Any location",
                            "technical": [],
                            "other": []
                        }
                    }
                    
                    # Get metadata and add it directly to the base profile
                    metadata = await self._generate_random_metadata(task, difficulty_level)
                    base_profile.update(metadata)
                    
                    # Modify prompt to explicitly request more ambiguous requirements
                    prompt = f"""Generate task-specific requirements and success criteria for:

Task: {task}
Base Profile: {json.dumps(base_profile, indent=2)}
Difficulty Level: {difficulty_level}
Option Number: {i + 1} of {num_options}

Requirements:
1. Generate a JSON object with the following structure:
{{
    "task_requirements": {{
        "technical": ["string"],
        "non_technical": ["string"]
    }},
    "success_criteria": {{
        "must_meet": ["string"],
        "should_meet": ["string"],
        "nice_to_meet": ["string"]
    }}
}}

2. IMPORTANT: Make this profile AMBIGUOUS based on difficulty level {difficulty_level}:
   - For difficulty 3+: Include vague requirements like "something modern" or "good performance"
   - For difficulty 4+: Add contradictory requirements
   - For difficulty 5: Make most requirements unclear, using phrases like "I think I need..."
   - Include more "Unknown/Not sure" entries at higher difficulties
   - Add statements that show knowledge gaps like "I heard X is important but I'm not sure why"
   - For technical requirements, use imprecise language that shows limited understanding

3. Express confusion about technical specifications - use incorrect terms or mix up concepts.

Write ONLY the JSON response. Do not include any explanations or additional text.
Do not include markdown code block markers (```json or ```)."""

                    # Create a ChatAgent for profile generation
                    agent = ChatAgent(
                        system_message="""You are a task profile generator for dialogue systems.
Your task is to generate realistic and specific task profiles with variations.
Always respond with valid JSON that matches the required structure.
Make each option different from others while maintaining realism.
Include "Unknown/Not sure" options where appropriate.
Do not include markdown code block markers in your response.""",
                        model=self.model
                    )
                    
                    # Generate requirements and criteria
                    response = agent.step(prompt)
                    content = self._extract_message_content(response)
                    
                    try:
                        # Try to parse the JSON response
                        additional_info = json.loads(content)
                        base_profile.update(additional_info)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing JSON response for option {i+1}: {str(e)}")
                        logger.error(f"Raw response content: {content}")
                        # If parsing fails, use default values
                        base_profile.update({
                            "task_requirements": {
                                "technical": ["Unknown/Not sure"],
                                "non_technical": ["Unknown/Not sure"]
                            },
                            "success_criteria": {
                                "must_meet": ["Unknown/Not sure"],
                                "should_meet": ["Unknown/Not sure"],
                                "nice_to_meet": ["Unknown/Not sure"]
                            }
                        })
                    
                    # Add metadata dictionary but keep required fields at top level
                    base_profile["metadata"] = {
                        "generation_timestamp": datetime.now().isoformat(),
                        "task": task,
                        "difficulty_level": difficulty_level,
                        "option_number": i + 1,
                        "total_options": num_options,
                        "model_type": self.model.model_type.name,
                        "uncertainty_score": min(10, difficulty_level * 2),  # Scale 1-10 based on difficulty
                        "is_uncertain": include_uncertain or difficulty_level > 3  # More profiles are uncertain
                    }
                    
                    # Validate and add to results
                    if self._validate_task_profile(base_profile):
                        task_profiles.append(base_profile)
                        logger.info(f"Successfully generated task profile option {i+1} for: {task}")
                    else:
                        logger.error(f"Generated profile for option {i+1} failed validation")
                    
                except Exception as e:
                    logger.error(f"Error generating task profile option {i+1}: {str(e)}")
                    continue
            
            if not task_profiles:
                raise ValueError("Failed to generate any valid task profiles")
            
            logger.info(f"Successfully generated {len(task_profiles)} task profile options for: {task}")
            return task_profiles
            
        except Exception as e:
            logger.error(f"Error generating task options: {str(e)}")
            raise

    def save_task_profile(self, profile: Dict[str, Any], filepath: str) -> bool:
        """
        Save the generated task profile to a file.
        
        Args:
            profile: The task profile to save
            filepath: Path to save the profile
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            # Validate profile before saving
            if not self._validate_task_profile(profile):
                logger.error("Invalid task profile, not saving")
                return False
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save profile with proper formatting
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(profile, f, indent=4, ensure_ascii=False)
            
            logger.info(f"Task profile saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving task profile: {str(e)}")
            return False

    def load_task_profile(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        Load a task profile from a file.
        
        Args:
            filepath: Path to the task profile file
            
        Returns:
            Optional[Dict[str, Any]]: Loaded profile if successful, None otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(filepath):
                logger.error(f"Task profile file not found: {filepath}")
                return None
            
            # Load profile
            with open(filepath, 'r', encoding='utf-8') as f:
                profile = json.load(f)
            
            # Validate loaded profile
            if not self._validate_task_profile(profile):
                logger.error("Invalid task profile loaded")
                return None
            
            logger.info(f"Task profile loaded from {filepath}")
            return profile
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing task profile file: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error loading task profile: {str(e)}")
            return None

    def get_all_task_profiles(self, directory: str) -> List[Dict[str, Any]]:
        """
        Get all task profiles from a directory.
        
        Args:
            directory: Directory containing task profile files
            
        Returns:
            List of task profiles
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
                    profile = self.load_task_profile(filepath)
                    if profile:
                        profiles.append(profile)
            
            logger.info(f"Loaded {len(profiles)} task profiles from {directory}")
            return profiles
            
        except Exception as e:
            logger.error(f"Error getting task profiles: {str(e)}")
            return []

    def _add_ambiguity(self, preferences: List[str], difficulty_level: int) -> List[str]:
        """Add ambiguity to preferences based on difficulty level."""
        if difficulty_level <= 2:
            return preferences
        
        vague_phrases = [
            "I think", "Maybe", "Probably", "I heard", "People say",
            "Supposedly", "Not sure but", "If possible"
        ]
        
        # Modify preferences to add ambiguity markers
        modified = []
        for pref in preferences:
            # Higher difficulty = higher chance of making items vague
            if random.random() < (difficulty_level * 0.15):
                prefix = random.choice(vague_phrases)
                modified.append(f"{prefix} {pref.lower()}")
            else:
                modified.append(pref)
            
        return modified 