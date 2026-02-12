"""
Quick test script to verify all components are working.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test if all required packages can be imported."""
    print("🧪 Testowanie importów...")
    
    try:
        import pandas
        print("  ✅ pandas")
    except ImportError as e:
        print(f"  ❌ pandas: {e}")
        
    try:
        import numpy
        print("  ✅ numpy")
    except ImportError as e:
        print(f"  ❌ numpy: {e}")
        
    try:
        import sklearn
        print("  ✅ scikit-learn")
    except ImportError as e:
        print(f"  ❌ scikit-learn: {e}")
        
    try:
        import xgboost
        print("  ✅ xgboost")
    except ImportError as e:
        print(f"  ❌ xgboost: {e}")
        
    try:
        import lightgbm
        print("  ✅ lightgbm")
    except ImportError as e:
        print(f"  ❌ lightgbm: {e}")
        
    try:
        import streamlit
        print("  ✅ streamlit")
    except ImportError as e:
        print(f"  ❌ streamlit: {e}")
        
    try:
        import openai
        print("  ✅ openai")
    except ImportError as e:
        print(f"  ❌ openai: {e}")
        
    try:
        import langfuse
        print("  ✅ langfuse")
    except ImportError as e:
        print(f"  ❌ langfuse: {e}")
        
    try:
        import boto3
        print("  ✅ boto3")
    except ImportError as e:
        print(f"  ❌ boto3: {e}")


def test_config():
    """Test configuration."""
    print("\n🔧 Testowanie konfiguracji...")
    
    try:
        import config
        print("  ✅ config.py załadowany")
        
        # Check essential config
        if config.DO_SPACES_CONFIG.get('aws_access_key_id'):
            print("  ⚠️  DO_SPACES_KEY jest ustawiony")
        else:
            print("  ⚠️  DO_SPACES_KEY nie jest ustawiony (wypełnij .env)")
            
        if config.OPENAI_API_KEY:
            print("  ✅ OPENAI_API_KEY jest ustawiony")
        else:
            print("  ⚠️  OPENAI_API_KEY nie jest ustawiony (wypełnij .env)")
            
        if config.LANGFUSE_PUBLIC_KEY:
            print("  ✅ LANGFUSE_PUBLIC_KEY jest ustawiony")
        else:
            print("  ⚠️  LANGFUSE_PUBLIC_KEY nie jest ustawiony (wypełnij .env)")
            
    except Exception as e:
        print(f"  ❌ Błąd konfiguracji: {e}")


def test_utils():
    """Test utility modules."""
    print("\n🛠️  Testowanie modułów pomocniczych...")
    
    try:
        from utils.spaces_handler import SpacesHandler
        print("  ✅ SpacesHandler")
    except Exception as e:
        print(f"  ❌ SpacesHandler: {e}")
        
    try:
        from utils.llm_extractor import DataExtractor
        print("  ✅ DataExtractor")
    except Exception as e:
        print(f"  ❌ DataExtractor: {e}")


def test_llm_extraction():
    """Test LLM extraction with a sample input."""
    print("\n🤖 Testowanie ekstrakcji LLM...")
    
    try:
        from utils.llm_extractor import extract_user_data
        
        test_input = "Jestem 30-letnim mężczyzną, 5km biegnę w 22:30"
        print(f"  Input: '{test_input}'")
        
        result = extract_user_data(test_input)
        print(f"  ✅ Wynik: {result}")
        
    except Exception as e:
        print(f"  ❌ Błąd: {e}")
        print(f"  💡 Upewnij się, że OPENAI_API_KEY jest poprawnie ustawiony")


def test_spaces_connection():
    """Test Digital Ocean Spaces connection."""
    print("\n☁️  Testowanie połączenia z Digital Ocean Spaces...")
    
    try:
        from utils.spaces_handler import SpacesHandler
        
        spaces = SpacesHandler()
        files = spaces.list_files(prefix="")
        
        print(f"  ✅ Połączono z Spaces")
        print(f"  📁 Znaleziono {len(files)} plików")
        
        if files:
            print("  Przykładowe pliki:")
            for file in files[:5]:
                print(f"    - {file}")
                
    except Exception as e:
        print(f"  ❌ Błąd: {e}")
        print(f"  💡 Sprawdź credentials w .env")


def main():
    """Run all tests."""
    print("=" * 60)
    print("🧪 HALF MARATHON PREDICTOR - TEST SUITE")
    print("=" * 60)
    
    test_imports()
    test_config()
    test_utils()
    
    # Optional tests (require API keys)
    print("\n" + "=" * 60)
    print("⚠️  TESTY WYMAGAJĄCE API KEYS")
    print("=" * 60)
    
    try:
        test_llm_extraction()
    except:
        print("  ⏭️  Pominięto test LLM (brak API key lub błąd)")
    
    try:
        test_spaces_connection()
    except:
        print("  ⏭️  Pominięto test Spaces (brak credentials lub błąd)")
    
    print("\n" + "=" * 60)
    print("✅ Testy zakończone!")
    print("=" * 60)


if __name__ == "__main__":
    main()
