"""
LLM Data Extractor with Langfuse monitoring
Wydobywa dane biegacza z naturalnego języka polskiego
"""
import os
import json
import streamlit as st
from openai import OpenAI

# Langfuse - import z try/except dla bezpieczeństwa
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    print("Warning: Langfuse not available")

SYSTEM_PROMPT = """Jesteś ekstraktor danych dla aplikacji biegowej.
Z tekstu użytkownika wydobądź:
  - gender  : "M" (mężczyzna) lub "K" (kobieta) lub null
  - age     : liczba całkowita (wiek w latach) lub null
  - time_5km: czas w formacie "MM:SS" lub null

Zasady:
- Wiek możesz wyliczyć z roku urodzenia (odejmij od 2024)
- Płeć wnioskuj z form gramatycznych / słów kluczowych
- Czas może być podany jako "22 minuty", "22:30", "22 min 30 sek" – znormalizuj do MM:SS
- Jeśli nie jesteś w stanie wydobyć wartości, użyj null
- Odpowiedz TYLKO i WYŁĄCZNIE poprawnym JSON, bez żadnego komentarza

Przykłady:
  "30-letni mężczyzna, 5km w 22:30"
  -> {"gender":"M","age":30,"time_5km":"22:30"}

  "kobieta, rocznik 1990, pięciokę biegam w 25 minut"
  -> {"gender":"K","age":34,"time_5km":"25:00"}

  "facet 45 lat"
  -> {"gender":"M","age":45,"time_5km":null}
"""


def extract_with_llm(user_text: str) -> dict:
    """
    Wywołaj OpenAI GPT-4o-mini i zmierz przez Langfuse.
    
    Args:
        user_text: Opis użytkownika w naturalnym języku
        
    Returns:
        dict: {gender: str, age: int, time_5km: str}
    """
    # Pobierz klucz OpenAI
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        st.error("Brak `OPENAI_API_KEY` – sprawdź zmienne środowiskowe.")
        return {}

    # ══════════════════════════════════════════════════════════
    # LANGFUSE MONITORING
    # ══════════════════════════════════════════════════════════
    langfuse_client = None
    trace = None
    generation = None
    
    if LANGFUSE_AVAILABLE:
        try:
            # Pobierz klucze Langfuse
            lf_pub = os.getenv("LANGFUSE_PUBLIC_KEY") or st.secrets.get("LANGFUSE_PUBLIC_KEY", "")
            lf_sec = os.getenv("LANGFUSE_SECRET_KEY") or st.secrets.get("LANGFUSE_SECRET_KEY", "")
            lf_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
            
            if lf_pub and lf_sec:
                # Inicjalizuj klienta Langfuse
                langfuse_client = Langfuse(
                    public_key=lf_pub,
                    secret_key=lf_sec,
                    host=lf_host
                )
                
                # Utwórz trace
                trace = langfuse_client.trace(
                    name="extract_runner_data",
                    input={"user_text": user_text},
                    metadata={"source": "streamlit_app"}
                )
                
                # Utwórz generation (przed wywołaniem OpenAI)
                generation = trace.generation(
                    name="gpt-4o-mini-extraction",
                    model="gpt-4o-mini",
                    input=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_text}
                    ],
                    metadata={"temperature": 0.0, "max_tokens": 150}
                )
                
        except Exception as e:
            print(f"Langfuse initialization error: {e}")
            # Kontynuuj bez Langfuse jeśli błąd

    # ══════════════════════════════════════════════════════════
    # OPENAI API CALL
    # ══════════════════════════════════════════════════════════
    try:
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=150,
        )
        
        # Parse response
        raw = response.choices[0].message.content
        result = json.loads(raw)
        
        # ══════════════════════════════════════════════════════════
        # LANGFUSE - Zapisz wynik
        # ══════════════════════════════════════════════════════════
        if generation:
            try:
                generation.end(
                    output=result,
                    usage={
                        "input": response.usage.prompt_tokens,
                        "output": response.usage.completion_tokens,
                        "total": response.usage.total_tokens
                    },
                    metadata={
                        "extracted_fields": list(result.keys()),
                        "missing_fields": [k for k, v in result.items() if v is None]
                    }
                )
            except Exception as e:
                print(f"Langfuse generation.end error: {e}")
        
        if langfuse_client:
            try:
                langfuse_client.flush()
            except Exception as e:
                print(f"Langfuse flush error: {e}")
        
        return result
        
    except Exception as e:
        error_msg = f"OpenAI API error: {str(e)}"
        st.error(error_msg)
        
        # Langfuse - zapisz błąd
        if generation:
            try:
                generation.end(
                    output={"error": error_msg},
                    level="ERROR"
                )
            except Exception:
                pass
                
        if langfuse_client:
            try:
                langfuse_client.flush()
            except Exception:
                pass
        
        return {}
