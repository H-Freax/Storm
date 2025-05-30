import random
from typing import Dict, Any, List
import json
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class RandomProfileGenerator:
    """
    A class for generating random user profiles from predefined pools.
    """

    # Predefined pools for different aspects of user profiles
    AGE_GROUPS = [
        "18-24", "25-34", "35-44", "45-54", "55-64", "65+"
    ]

    TECH_EXPERIENCE = [
        "Expert", "Advanced", "Intermediate", "Beginner", "Novice"
    ]

    LANGUAGE_STYLES = [
        "Formal", "Casual", "Technical", "Simple", "Professional"
    ]

    PERSONALITIES = [
        "Friendly", "Reserved", "Outgoing", "Analytical", "Creative"
    ]

    CULTURES = [
        "Western", "Eastern", "Middle Eastern", "African", "Latin American"
    ]

    DECISION_STYLES = [
        "Rational", "Intuitive", "Cautious", "Impulsive", "Balanced"
    ]

    COMMUNICATION_STYLES = [
        "Direct", "Indirect", "Detailed", "Concise", "Adaptive"
    ]

    EXPRESSIVENESS = [
        "Very Expressive", "Moderately Expressive", "Neutral", "Reserved", "Very Reserved"
    ]

    SOCIAL_CONTEXTS = [
        "Professional", "Personal", "Academic", "Social", "Mixed"
    ]

    PHYSICAL_STATUS = [
        "Active", "Sedentary", "Limited Mobility", "Athletic", "Average"
    ]

    # Predefined pools for behavioral traits
    PATIENCE_LEVELS = [
        "Very Patient", "Patient", "Moderate", "Impatient", "Very Impatient"
    ]

    ATTENTION_TO_DETAIL = [
        "Very Detailed", "Detailed", "Moderate", "Basic", "Minimal"
    ]

    RISK_TOLERANCE = [
        "Very Risk-Averse", "Risk-Averse", "Moderate", "Risk-Taking", "Very Risk-Taking"
    ]

    ADAPTABILITY = [
        "Very Adaptable", "Adaptable", "Moderate", "Resistant", "Very Resistant"
    ]

    LEARNING_STYLES = [
        "Visual", "Auditory", "Reading/Writing", "Kinesthetic", "Mixed"
    ]

    # Predefined pools for contextual factors
    TIME_CONSTRAINTS = [
        "Very Urgent", "Urgent", "Moderate", "Flexible", "Very Flexible"
    ]

    ENVIRONMENTS = [
        "Home", "Office", "Public Space", "Mobile", "Mixed"
    ]

    SOCIAL_PRESSURES = [
        "High", "Moderate", "Low", "None", "Mixed"
    ]

    PREVIOUS_EXPERIENCE = [
        "Extensive", "Moderate", "Limited", "None", "Mixed"
    ]

    # Predefined example dimensions for different difficulty levels
    EXAMPLE_DIMENSIONS = {
        "1": {
            "style": "comprehensive_and_detailed",
            "length": "extended",
            "content": "complete_needs_with_context",
            "tone": "enthusiastic_and_specific",
            "examples": [
                "I'm looking for a 5G smartphone under $800 with at least 8GB RAM, excellent camera quality, and all-day battery life since I travel frequently for work and need to take lots of photos.",
                "I need a gaming laptop with an RTX 3070 or better GPU, 32GB RAM, and a 144Hz display that can handle modern AAA titles at high settings while staying cool during extended sessions.",
                "I want a smart home security system that includes doorbell camera, 4 outdoor cameras, motion sensors, and integrates with Google Home since we have a large property and travel frequently."
            ]
        },
        "2": {
            "style": "clear_but_selective",
            "length": "moderate",
            "content": "primary_needs_with_some_context",
            "tone": "friendly_but_businesslike",
            "examples": [
                "I need a smartphone with a good camera and decent battery life. Something reliable for everyday use.",
                "Looking for a gaming laptop that can handle new releases. I'd prefer something with a good display.",
                "Can you recommend a home security system? We have a medium-sized house and would need outdoor coverage."
            ]
        },
        "3": {
            "style": "basic_and_direct",
            "length": "brief",
            "content": "core_need_only",
            "tone": "neutral",
            "examples": [
                "I need a new smartphone. What do you recommend?",
                "Looking for a laptop for gaming. Options?",
                "Need a home security system. What's available?"
            ]
        },
        "4": {
            "style": "vague_and_minimal",
            "length": "very_brief",
            "content": "implied_needs",
            "tone": "detached",
            "examples": [
                "Need a phone soon.",
                "Computer for games?",
                "House security."
            ]
        },
        "5": {
            "style": "cryptic_or_misleading",
            "length": "minimal",
            "content": "ambiguous_or_contradictory",
            "tone": "confusing_or_frustrated",
            "examples": [
                "Something in my pocket broke.",
                "Screen thing too slow games.",
                "Don't feel safe sometimes maybe."
            ]
        }
    }

    # Predefined difficulty instructions
    DIFFICULTY_INSTRUCTIONS = {
        "dialogue": {
            "1": "Share your needs and preferences clearly.",
            "2": "Share some key information but keep some details private.",
            "3": "Express only basic needs, let the assistant ask questions.",
            "4": "Be brief and direct, give minimal information.",
            "5": "Use vague or ambiguous language about your needs."
        },
        "profile": {
            "1": "Reveal all your profile information naturally in the conversation.",
            "2": "Share most of your profile information when asked.",
            "3": "Gradually reveal your profile information throughout the conversation.",
            "4": "Only share profile information when specifically asked.",
            "5": "Be very reluctant to share your profile information."
        },
        "hidden_state": {
            "1": "Express strong emotions and clear intentions in your inner thoughts.",
            "2": "Express clear emotions and intentions in your inner thoughts.",
            "3": "Express moderate emotions and intentions in your inner thoughts.",
            "4": "Express minimal emotions and intentions in your inner thoughts.",
            "5": "Express very subtle emotions and intentions in your inner thoughts."
        }
    }

    def __init__(self):
        """Initialize the random profile generator."""
        pass

    def generate_profile(self) -> Dict[str, Any]:
        """
        Generate a random user profile from predefined pools.
        
        Returns:
            Dict containing the generated user profile
        """
        try:
            # Generate base profile
            base_profile = {
                "age_group": random.choice(self.AGE_GROUPS),
                "tech_experience": random.choice(self.TECH_EXPERIENCE),
                "language_style": random.choice(self.LANGUAGE_STYLES),
                "personality": random.choice(self.PERSONALITIES),
                "culture": random.choice(self.CULTURES),
                "decision_style": random.choice(self.DECISION_STYLES),
                "communication_style": random.choice(self.COMMUNICATION_STYLES),
                "expressiveness": random.choice(self.EXPRESSIVENESS),
                "social_context": random.choice(self.SOCIAL_CONTEXTS),
                "physical_status": random.choice(self.PHYSICAL_STATUS)
            }

            # Generate behavioral traits
            behavioral_traits = {
                "patience": random.choice(self.PATIENCE_LEVELS),
                "attention_to_detail": random.choice(self.ATTENTION_TO_DETAIL),
                "risk_tolerance": random.choice(self.RISK_TOLERANCE),
                "adaptability": random.choice(self.ADAPTABILITY),
                "learning_style": random.choice(self.LEARNING_STYLES)
            }

            # Generate contextual factors
            contextual_factors = {
                "time_constraint": random.choice(self.TIME_CONSTRAINTS),
                "environment": random.choice(self.ENVIRONMENTS),
                "social_pressure": random.choice(self.SOCIAL_PRESSURES),
                "previous_experience": random.choice(self.PREVIOUS_EXPERIENCE)
            }

            # Create the complete profile
            profile = {
                "name": f"User_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "description": "Randomly generated user profile",
                "base_profile": base_profile,
                "behavioral_traits": behavioral_traits,
                "contextual_factors": contextual_factors
            }

            logger.info("Successfully generated random user profile")
            return profile

        except Exception as e:
            logger.error(f"Error generating random profile: {str(e)}")
            raise

    def get_example_dimensions(self, difficulty_level: int) -> Dict[str, Any]:
        """
        Get example dimensions for a specific difficulty level.
        
        Args:
            difficulty_level: The difficulty level (1-5)
            
        Returns:
            Dict containing example dimensions
        """
        return self.EXAMPLE_DIMENSIONS.get(str(difficulty_level), self.EXAMPLE_DIMENSIONS["3"])

    def get_difficulty_instructions(self, difficulty_level: int) -> Dict[str, str]:
        """
        Get difficulty instructions for a specific difficulty level.
        
        Args:
            difficulty_level: The difficulty level (1-5)
            
        Returns:
            Dict containing difficulty instructions
        """
        return {
            "dialogue": self.DIFFICULTY_INSTRUCTIONS["dialogue"].get(str(difficulty_level), self.DIFFICULTY_INSTRUCTIONS["dialogue"]["3"]),
            "profile": self.DIFFICULTY_INSTRUCTIONS["profile"].get(str(difficulty_level), self.DIFFICULTY_INSTRUCTIONS["profile"]["3"]),
            "hidden_state": self.DIFFICULTY_INSTRUCTIONS["hidden_state"].get(str(difficulty_level), self.DIFFICULTY_INSTRUCTIONS["hidden_state"]["3"])
        } 