"""
Half Marathon Time Predictor — Streamlit App
=============================================
Użytkownik opisuje siebie po polsku, LLM wyciąga
(płeć, wiek, czas_5km), model ML predykuje czas półmaratonu.
"""
import os, json, re
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import joblib

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="🏃 Kalkulator półmaratonu",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def seconds_to_hhmmss(s: float) -> str:
    s = max(0, int(s))
    return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"


def time_str_to_seconds(t: str) -> int | None:
    """MM:SS lub HH:MM:SS -> sekundy."""
    try:
        parts = t.strip().split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except Exception:
        return None


# ──────────────────────────────────────────────
# MODEL LOADER
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Wczytywanie modelu…")
def load_model():
    """Wczytaj model – najpierw lokalnie, potem z Spaces."""
    local = Path(__file__).parent / "models" / "halfmarathon_model.pkl"

    if not local.exists():
        st.info("📥 Pobieranie modelu z Digital Ocean Spaces…")
        try:
            from utils.spaces_handler import download_model
            local.parent.mkdir(exist_ok=True)
            download_model("halfmarathon_model.pkl", str(local))
        except Exception as e:
            st.error(f"Nie można pobrać modelu: {e}")
            return None

    if not local.exists():
        st.error("Brak pliku modelu. Uruchom najpierw: `python train_quick.py --local`")
        return None

    return joblib.load(local)


# ──────────────────────────────────────────────
# LLM EXTRACTOR  (openai + langfuse)
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """Jesteś ekstraktor danych dla aplikacji biegowej.
Z tekstu użytkownika wydobądź:
  - gender  : "M" (mężczyzna) lub "K" (kobieta) lub null
  - age     : liczba całkowita (wiek w latach) lub null
  - time_5km: czas w formacie "MM:SS" lub null

Zasady:
• Wiek możesz wyliczyć z roku urodzenia (odejmij od 2024)
• Płeć wnioskuj z form gramatycznych / słów kluczowych
• Czas może być podany jako "22 minuty", "22:30", "22 min 30 sek" – znormalizuj do MM:SS
• Jeśli nie jesteś w stanie wydobyć wartości, użyj null
• Odpowiedz TYLKO i WYŁĄCZNIE poprawnym JSON, bez żadnego komentarza

Przykłady:
  "30-letni mężczyzna, 5km w 22:30"
  -> {"gender":"M","age":30,"time_5km":"22:30"}

  "kobieta, rocznik 1990, pięciokę biegam w 25 minut"
  -> {"gender":"K","age":34,"time_5km":"25:00"}

  "facet 45 lat"
  -> {"gender":"M","age":45,"time_5km":null}
"""


def extract_with_llm(user_text: str) -> dict:
    """Wywołaj OpenAI i zmierz przez Langfuse."""
    
    # ── OpenAI API key ────────────────────────────
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        st.error("Brak `OPENAI_API_KEY` – sprawdź plik .env lub Streamlit Secrets.")
        return {}

    # ── Langfuse (NOWE API v3) ────────────────────
    lf_pub = os.getenv("LANGFUSE_PUBLIC_KEY") or st.secrets.get("LANGFUSE_PUBLIC_KEY", "")
    lf_sec = os.getenv("LANGFUSE_SECRET_KEY") or st.secrets.get("LANGFUSE_SECRET_KEY", "")
    
    try:
        if lf_pub and lf_sec:
            from langfuse.openai import openai
            client = openai.OpenAI(api_key=api_key)
            st.success("✅ Langfuse tracking aktywny!")
        else:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            st.warning("⚠️ Brak kluczy Langfuse - tracking wyłączony")
    except Exception as e:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        st.warning(f"⚠️ Langfuse niedostępny: {e}")

    # ── OpenAI call ───────────────────────────────
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

    raw = response.choices[0].message.content
    result = json.loads(raw)
    
    return result
 # ──────────────────────────────────────────────
# PREDICTION
# ──────────────────────────────────────────────
def predict(pkg: dict, age: int, gender: str, time_5km_s: int) -> float:
    """Zwraca przewidywany czas półmaratonu w sekundach."""
    le    = pkg["label_encoder"]
    model = pkg["model"]
    g_enc = le.transform([gender])[0]

    X = pd.DataFrame(
        [[age, g_enc, time_5km_s]],
        columns=pkg["features"],
    )
    return float(model.predict(X)[0])


# ──────────────────────────────────────────────
# UI HELPERS
# ──────────────────────────────────────────────
def metric_card(label: str, value: str, icon: str = ""):
    st.markdown(f"""
    <div style="background:#f0f4ff;border-radius:12px;padding:18px 20px;text-align:center;">
        <div style="font-size:2rem;">{icon}</div>
        <div style="font-size:1.7rem;font-weight:700;color:#1a3c8f;">{value}</div>
        <div style="font-size:0.85rem;color:#555;margin-top:4px;">{label}</div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    # ---- Header ----
    st.markdown("""
    <h1 style="text-align:center;color:#1a3c8f;margin-bottom:0;">🏃 Kalkulator Czasu Półmaratonu</h1>
    <p style="text-align:center;color:#666;font-size:1.1rem;margin-top:4px;">
        Powered by AI · Model trenowany na 18 000+ wynikach z Wrocławia 2023-2024
    </p>
    """, unsafe_allow_html=True)

    st.divider()

    # ---- Load model ----
    pkg = load_model()
    if pkg is None:
        st.stop()

    meta = pkg.get("metadata", {})

    # ---- Sidebar – info o modelu ----
    # ---- Sidebar – info o modelu ----
    with st.sidebar:
        st.header("ℹ️ O modelu")
        st.metric("Typ modelu",   meta.get("model_name", "—"))
        st.metric("MAE",          f"{meta.get('mae_seconds', 0)/60:.2f} min")
        st.metric("RMSE",         f"{meta.get('rmse_seconds', 0)/60:.2f} min")
        st.metric("R²",           f"{meta.get('r2', 0):.4f}")
        
        train_samples = meta.get('train_samples', 0)
        if train_samples and isinstance(train_samples, (int, float)):
            st.metric("Próbek train", f"{int(train_samples):,}")
        else:
            st.metric("Próbek train", "—")
        
        if meta.get("training_date"):
            d = datetime.fromisoformat(meta["training_date"])
            st.metric("Trening",  d.strftime("%Y-%m-%d %H:%M"))

        st.divider()
        st.caption("Cechy wejściowe: wiek, płeć, czas na 5km  \n"
                   "Dane: Półmaraton Wrocław 2023-2024")

### **OPCJA 4: Cofnij zmiany**

    

    # ---- Główna kolumna ----
    col_main, col_hist = st.columns([3, 2], gap="large")

    with col_main:
        st.subheader("📝 Przedstaw się")

        with st.expander("💡 Przykłady wpisów"):
            st.markdown("""
| Przykład |
|---|
| `30-letni mężczyzna, 5km biegnę w 22:30` |
| `Kobieta, 25 lat, czas na 5km to 25 minut` |
| `45-letni facet, pięciokę robię w 20:00` |
| `Rocznik 1990, płeć żeńska, 5km w 28 min 15 sek` |
| `Mam 35 lat, jestem kobietą i 5km pokonuję w 27:45` |
""")

        user_text = st.text_area(
            "Opisz siebie (płeć, wiek, czas na 5km):",
            placeholder="np. Mam 30 lat, jestem mężczyzną, 5km biegnę w 22:30",
            height=90,
            key="user_input",
        )

        go = st.button("🔮 Oblicz mój czas półmaratonu", type="primary", use_container_width=True)

    # ---- Historia ----
    if "history" not in st.session_state:
        st.session_state.history = []

    with col_hist:
        st.subheader("📜 Historia predykcji")
        if st.session_state.history:
            hist_df = pd.DataFrame(st.session_state.history)[
                ["Płeć", "Wiek", "5km", "Prognoza"]
            ]
            st.dataframe(hist_df, use_container_width=True, hide_index=True)
            if st.button("🗑️ Wyczyść", use_container_width=True):
                st.session_state.history = []
                st.rerun()
        else:
            st.info("Twoje predykcje pojawią się tutaj.")

    # ──────────────────────────────────────────────
    # MAIN LOGIC
    # ──────────────────────────────────────────────
    if not go:
        return

    if not user_text.strip():
        st.warning("⚠️ Wpisz coś w polu tekstowym!")
        return

    st.divider()

    # ---- Extract ----
    with st.spinner("🤖 Analizuję tekst za pomocą AI…"):
        try:
            data = extract_with_llm(user_text)
        except Exception as e:
            st.error(f"Błąd podczas wywoływania LLM: {e}")
            st.info("Sprawdź czy `OPENAI_API_KEY` jest poprawnie ustawiony.")
            return

    # ---- Wyświetl wydobyte dane ----
    st.subheader("🔍 Wydobyte dane")
    c1, c2, c3 = st.columns(3)

    gender_display = {"M": "👨 Mężczyzna", "K": "👩 Kobieta", None: "❓ —"}

    with c1:
        metric_card("Płeć", gender_display.get(data.get("gender"), "❓ —"), "")
    with c2:
        age = data.get("age")
        metric_card("Wiek", f"{age} lat" if age else "—", "🎂")
    with c3:
        t5 = data.get("time_5km")
        metric_card("Czas 5km", t5 or "—", "⏱️")

    # ---- Walidacja ----
    missing = []
    if not data.get("gender"):      missing.append("**płeć**")
    if not data.get("age"):         missing.append("**wiek**")
    if not data.get("time_5km"):    missing.append("**czas na 5km**")

    if missing:
        st.warning(f"⚠️ Brakuje: {', '.join(missing)}. "
                   f"Uzupełnij opis i spróbuj ponownie.")
        return

    # ---- Predict ----
    t5km_s = time_str_to_seconds(data["time_5km"])
    if t5km_s is None:
        st.error("Nie rozpoznano czasu na 5km. Użyj formatu MM:SS, np. `22:30`.")
        return

    pred_s   = predict(pkg, int(data["age"]), data["gender"], t5km_s)
    pred_str = seconds_to_hhmmss(pred_s)

    pace_per_km_s = pred_s / 21.097
    pace_str = f"{int(pace_per_km_s//60)}:{int(pace_per_km_s%60):02d} min/km"

    # ---- Wynik ----
    st.divider()
    st.subheader("🎉 Twój przewidywany czas półmaratonu")

    st.markdown(f"""
    <div style="text-align:center;padding:30px 20px;
                background:linear-gradient(135deg,#1a3c8f,#0066cc);
                border-radius:16px;color:white;margin:10px 0;">
        <div style="font-size:4rem;font-weight:900;letter-spacing:4px;">{pred_str}</div>
        <div style="font-size:1.1rem;opacity:0.85;margin-top:6px;">gg:mm:ss</div>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        metric_card("Tempo",          pace_str,                       "🏃")
    with m2:
        metric_card("Dystans",        "21.097 km",                    "📍")
    with m3:
        metric_card("Dokładność modelu", f"± {meta.get('mae_seconds',0)/60:.1f} min", "🎯")

    # ---- Wskazówka treningowa ----
    t5km_pace = t5km_s / 5.0         # s/km przy 5km
    target_pace = pred_s / 21.097     # s/km cel
    diff = target_pace - t5km_pace    # >0: spowalniasz na dłuższym

    st.divider()
    if diff > 120:
        st.info("💡 **Dużo tracisz na dłuższym dystansie** – skup się na wytrwałości: "
                "długie biegi w weekendy, budowanie bazy aerobowej.")
    elif diff > 60:
        st.info("💡 **Wytrzymałość do poprawy** – dodaj biegi progowe (tempo runs) "
                "i powoli wydłużaj dystans długiego biegu.")
    elif diff > 20:
        st.success("👍 **Dobry wynik!** Twoje tempo jest dość stabilne. "
                   "Pracuj nad finiszem – ćwicz ujemny podział (negative split).")
    else:
        st.success("🔥 **Świetna wytrzymałość!** Twoje tempo długiego biegu "
                   "jest zbliżone do 5km. Możesz pobić PR – zaplanuj start!")

    # ---- Zapis historii ----
    st.session_state.history.insert(0, {
        "Płeć":    data["gender"],
        "Wiek":    data["age"],
        "5km":     data["time_5km"],
        "Prognoza": pred_str,
        "_ts":     datetime.now().isoformat(),
    })

    # ---- Pełny wynik w expanderze ----
    with st.expander("🔧 Szczegóły predykcji"):
        st.json({
            "input": {
                "gender":    data["gender"],
                "age":       data["age"],
                "time_5km":  data["time_5km"],
                "time_5km_seconds": t5km_s,
            },
            "output": {
                "predicted_seconds": round(pred_s),
                "predicted_time":    pred_str,
                "pace_per_km":       pace_str,
            },
            "model_metadata": {
                "name": meta.get("model_name"),
                "mae_min": round(meta.get("mae_seconds", 0) / 60, 2),
                "r2": round(meta.get("r2", 0), 4),
            }
        })


if __name__ == "__main__":
    main()
