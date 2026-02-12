"""
Test modelu - bez LLM, bez API
Prosty test czy model działa i generuje predykcje.
"""
import joblib
import pandas as pd

print("="*60)
print(" TEST MODELU - Marathon Predictor")
print("="*60)

# Wczytaj model
try:
    pkg = joblib.load('models/halfmarathon_model.pkl')
    print("\n✅ Model wczytany pomyślnie!")
except FileNotFoundError:
    print("\n❌ BŁĄD: Nie znaleziono modelu w models/halfmarathon_model.pkl")
    print("   Sprawdź czy plik istnieje!")
    exit(1)

# Info o modelu
meta = pkg['metadata']
print(f"\n📊 Informacje o modelu:")
print(f"   Typ:        {meta['model_name']}")
print(f"   MAE:        {meta['mae_seconds']/60:.2f} min")
print(f"   RMSE:       {meta['rmse_seconds']/60:.2f} min")
print(f"   R²:         {meta['r2']:.4f}")
print(f"   Cechy:      {', '.join(meta['features'])}")

# Funkcja predykcji
def predict(age: int, gender: str, time_5km_str: str):
    """Przewiduje czas półmaratonu."""
    parts = time_5km_str.split(':')
    time_5km_s = int(parts[0]) * 60 + int(parts[1])
    
    le = pkg['label_encoder']
    gender_enc = le.transform([gender])[0]
    
    X = pd.DataFrame([[age, gender_enc, time_5km_s]], columns=pkg['features'])
    pred_s = pkg['model'].predict(X)[0]
    
    h = int(pred_s // 3600)
    m = int((pred_s % 3600) // 60)
    s = int(pred_s % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# Testy
print("\n🧪 Przykładowe predykcje:")
print("   (bez LLM - dane wpisane ręcznie)\n")

test_cases = [
    (30, 'M', '22:30', "Typowy mężczyzna amator"),
    (25, 'K', '25:00', "Młoda kobieta"),
    (45, 'M', '20:00', "Doświadczony zawodnik"),
    (35, 'K', '28:00', "Rekreacyjna biegaczka"),
    (50, 'M', '30:00', "Senior"),
    (22, 'K', '22:00', "Bardzo dobra młoda zawodniczka"),
]

for age, gender, t5km, opis in test_cases:
    result = predict(age, gender, t5km)
    gender_pl = "Mężczyzna" if gender == "M" else "Kobieta"
    print(f"   {gender_pl:10s} {age:2d} lat, 5km={t5km}  →  {result}")
    print(f"   ({opis})")
    print()

print("="*60)
print(" TEST ZAKOŃCZONY POMYŚLNIE!")
print("="*60)
print("\n💡 Następny krok: Uruchom aplikację (streamlit run app.py)")
print("   aby testować z LLM i interfejsem webowym.\n")
