"""
LLM-based data extractor with Langfuse integration.
Extracts structured data from user's natural language input.
"""
import json
import logging
from typing import Dict, Optional, List
from openai import OpenAI
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataExtractor:
    """Extract structured data from natural language using LLM."""
    
    def __init__(self):
        """Initialize OpenAI client and Langfuse."""
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        
        # Initialize Langfuse
        if config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY:
            self.langfuse = Langfuse(
                public_key=config.LANGFUSE_PUBLIC_KEY,
                secret_key=config.LANGFUSE_SECRET_KEY,
                host=config.LANGFUSE_HOST,
            )
        else:
            self.langfuse = None
            logger.warning("Langfuse not configured. Metrics will not be collected.")
    
    @observe(as_type="generation")
    def extract_data(self, user_input: str) -> Dict:
        """
        Extract structured data from user input.
        
        Args:
            user_input: Natural language input from user
            
        Returns:
            Dictionary with extracted data and validation info
        """
        system_prompt = """Jesteś asystentem do ekstrakcji danych o biegaczach. 
Twoim zadaniem jest wydobyć z tekstu użytkownika następujące informacje:
- płeć (gender): "M" dla mężczyzn, "K" dla kobiet
- wiek (age): liczba całkowita
- czas na 5km (time_5km): w formacie MM:SS lub HH:MM:SS

Odpowiedz TYLKO w formacie JSON z następującą strukturą:
{
    "gender": "M" lub "K" lub null,
    "age": liczba lub null,
    "time_5km": "MM:SS" lub null,
    "missing_fields": lista brakujących pól,
    "confidence": "high" lub "medium" lub "low"
}

Przykłady:
- "Jestem 30-letnim mężczyzną, 5km biegnę w 22:30" -> {"gender": "M", "age": 30, "time_5km": "22:30", "missing_fields": [], "confidence": "high"}
- "Kobieta, 25 lat" -> {"gender": "K", "age": 25, "time_5km": null, "missing_fields": ["time_5km"], "confidence": "high"}
- "Biegnę 5km w 20 minut" -> {"gender": null, "age": null, "time_5km": "20:00", "missing_fields": ["gender", "age"], "confidence": "medium"}

Zasady:
1. Jeśli informacja nie jest podana, użyj null
2. Wiek możesz wyliczyć z roku urodzenia (odejmij od 2024)
3. Płeć możesz wywnioskować z form gramatycznych (jestem, biegnę itp.)
4. Czas może być podany w różnych formatach - normalizuj do MM:SS
5. Lista missing_fields powinna zawierać tylko te pola, których nie udało się wydobyć
6. Confidence: high jeśli dane są jasno podane, medium jeśli wywnioskowałeś, low jeśli niepewny
"""

        try:
            # Call OpenAI API with Langfuse observation
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            # Extract and parse response
            result = json.loads(response.choices[0].message.content)
            
            # Log to Langfuse
            if self.langfuse:
                langfuse_context.update_current_observation(
                    input=user_input,
                    output=result,
                    metadata={
                        "model": "gpt-4o-mini",
                        "missing_fields": result.get("missing_fields", []),
                        "confidence": result.get("confidence", "unknown")
                    }
                )
            
            logger.info(f"Successfully extracted data: {result}")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return self._get_empty_result(["gender", "age", "time_5km"])
            
        except Exception as e:
            logger.error(f"Error in data extraction: {e}")
            return self._get_empty_result(["gender", "age", "time_5km"])
    
    def _get_empty_result(self, missing_fields: List[str]) -> Dict:
        """Return an empty result with all fields missing."""
        return {
            "gender": None,
            "age": None,
            "time_5km": None,
            "missing_fields": missing_fields,
            "confidence": "low"
        }
    
    def validate_extracted_data(self, data: Dict) -> Dict:
        """
        Validate extracted data and provide user-friendly messages.
        
        Args:
            data: Extracted data dictionary
            
        Returns:
            Dictionary with validation results
        """
        missing = data.get("missing_fields", [])
        
        if not missing:
            return {
                "is_valid": True,
                "message": "Wszystkie dane zostały pomyślnie wydobyte!",
                "missing_fields": []
            }
        
        # Create user-friendly message
        field_names = {
            "gender": "płeć",
            "age": "wiek",
            "time_5km": "czas na 5km"
        }
        
        missing_names = [field_names.get(field, field) for field in missing]
        message = f"Brakuje następujących danych: {', '.join(missing_names)}. "
        message += "Proszę podać te informacje, aby otrzymać prognozę."
        
        return {
            "is_valid": False,
            "message": message,
            "missing_fields": missing
        }
    
    def convert_time_to_seconds(self, time_str: str) -> Optional[int]:
        """
        Convert time string (MM:SS or HH:MM:SS) to seconds.
        
        Args:
            time_str: Time in format MM:SS or HH:MM:SS
            
        Returns:
            Time in seconds or None if invalid
        """
        if not time_str:
            return None
            
        try:
            parts = time_str.split(":")
            if len(parts) == 2:  # MM:SS
                minutes, seconds = map(int, parts)
                return minutes * 60 + seconds
            elif len(parts) == 3:  # HH:MM:SS
                hours, minutes, seconds = map(int, parts)
                return hours * 3600 + minutes * 60 + seconds
            else:
                return None
        except (ValueError, AttributeError):
            return None


def extract_user_data(user_input: str) -> Dict:
    """
    Convenience function to extract data from user input.
    
    Args:
        user_input: Natural language input from user
        
    Returns:
        Dictionary with extracted and validated data
    """
    extractor = DataExtractor()
    extracted = extractor.extract_data(user_input)
    validation = extractor.validate_extracted_data(extracted)
    
    return {
        **extracted,
        "validation": validation
    }
