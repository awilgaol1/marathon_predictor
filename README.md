# 🏃 Half Marathon Time Predictor

Aplikacja do przewidywania czasu ukończenia półmaratonu na podstawie wieku, płci i czasu na dystansie 5 km. Wykorzystuje Machine Learning i Large Language Models do ekstrakcji danych z naturalnego języka.

## 🎯 Funkcjonalności

- **Ekstrakcja danych z LLM**: Użytkownik opisuje się w naturalnym języku, a AI wydobywa potrzebne informacje
- **Predykcja ML**: Model trenowany na rzeczywistych danych z półmaratonów przewiduje czas ukończenia
- **Monitoring Langfuse**: Wszystkie wywołania LLM są monitorowane i analizowane
- **Digital Ocean Spaces**: Przechowywanie danych i modeli w chmurze
- **Streamlit UI**: Przyjazny interfejs użytkownika

## 🏗️ Architektura

```
halfmarathon_predictor/
├── app.py                      # Aplikacja Streamlit
├── config.py                   # Konfiguracja
├── requirements.txt            # Zależności
├── upload_data.py             # Skrypt do uploadu danych
├── Dockerfile                  # Konfiguracja Docker
├── .env.example               # Przykładowa konfiguracja
├── utils/
│   ├── __init__.py
│   ├── spaces_handler.py      # Obsługa Digital Ocean Spaces
│   └── llm_extractor.py       # Ekstrakcja danych przez LLM
├── notebooks/
│   └── train_model.ipynb      # Pipeline treningowy
├── data/                       # Dane (lokalne)
└── models/                     # Modele (lokalne)
```

## 📋 Wymagania

- Python 3.10+
- Konto Digital Ocean (dla Spaces)
- Konto OpenAI (dla API)
- Konto Langfuse (dla monitoringu)

## 🚀 Szybki start

### 1. Klonowanie i instalacja

```bash
# Sklonuj repozytorium
git clone <your-repo-url>
cd halfmarathon_predictor

# Utwórz wirtualne środowisko
python -m venv venv
source venv/bin/activate  # Na Windows: venv\Scripts\activate

# Zainstaluj zależności
pip install -r requirements.txt
```

### 2. Konfiguracja środowiska

Skopiuj `.env.example` do `.env` i wypełnij danymi:

```bash
cp .env.example .env
```

Edytuj `.env`:

```env
# Digital Ocean Spaces
DO_SPACES_REGION=fra1
DO_SPACES_ENDPOINT=https://fra1.digitaloceanspaces.com
DO_SPACES_KEY=your_access_key
DO_SPACES_SECRET=your_secret_key
DO_SPACES_BUCKET=halfmarathon-predictor

# OpenAI
OPENAI_API_KEY=sk-your-api-key

# Langfuse
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key
LANGFUSE_SECRET_KEY=sk-lf-your-secret-key
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 3. Konfiguracja Digital Ocean Spaces

#### Tworzenie Spaces Bucket

1. Zaloguj się do Digital Ocean
2. Przejdź do **Spaces** → **Create Space**
3. Wybierz region (np. Frankfurt - `fra1`)
4. Nazwa: `halfmarathon-predictor`
5. Włącz CDN (opcjonalnie)
6. Kliknij **Create Space**

#### Generowanie API Keys

1. Przejdź do **API** → **Spaces Keys**
2. Kliknij **Generate New Key**
3. Skopiuj **Access Key** i **Secret Key**
4. Wklej do pliku `.env`

### 4. Upload danych do Spaces

```bash
# Upewnij się, że pliki CSV są dostępne
# Zaktualizuj ścieżki w upload_data.py jeśli potrzeba

python upload_data.py
```

### 5. Trenowanie modelu

Otwórz i uruchom notebook:

```bash
jupyter notebook notebooks/train_model.ipynb
```

Notebook wykonuje następujące kroki:
1. ✅ Pobiera dane z Digital Ocean Spaces
2. ✅ Czyści i przygotowuje dane
3. ✅ Trenuje i porównuje różne modele
4. ✅ Wybiera najlepszy model
5. ✅ Zapisuje model lokalnie i w Spaces

### 6. Uruchomienie aplikacji lokalnie

```bash
streamlit run app.py
```

Aplikacja będzie dostępna pod adresem: `http://localhost:8501`

## 🌐 Deployment na Digital Ocean App Platform

### Metoda 1: Przez GitHub (Zalecana)

1. **Pushuj kod do GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Utwórz App w Digital Ocean**:
   - Przejdź do **App Platform** → **Create App**
   - Wybierz źródło: **GitHub**
   - Wybierz repozytorium
   - Branch: `main`

3. **Konfiguracja App**:
   - **Type**: Web Service
   - **Build Command**: (puste - używamy Dockerfile)
   - **Run Command**: (puste - używamy Dockerfile)
   - **HTTP Port**: `8501`
   - **Instance Size**: Basic ($5/mo)

4. **Dodaj zmienne środowiskowe**:
   - Przejdź do **Settings** → **Environment Variables**
   - Dodaj wszystkie zmienne z pliku `.env`
   - ⚠️ **Ważne**: NIE commituj pliku `.env` do repo!

5. **Deploy**:
   - Kliknij **Create Resources**
   - Poczekaj na deployment (~5-10 minut)
   - Twoja aplikacja będzie dostępna pod adresem `*.ondigitalocean.app`

### Metoda 2: Docker Container Registry

```bash
# Build image
docker build -t halfmarathon-predictor .

# Tag and push to DO Registry
doctl registry login
docker tag halfmarathon-predictor registry.digitalocean.com/<your-registry>/halfmarathon-predictor
docker push registry.digitalocean.com/<your-registry>/halfmarathon-predictor
```

## 🧪 Testowanie

### Test LLM Extractor

```python
from utils.llm_extractor import extract_user_data

result = extract_user_data("Jestem 30-letnim mężczyzną, 5km biegnę w 22:30")
print(result)
```

### Test Model Prediction

```python
import joblib
import pandas as pd

model_package = joblib.load('models/halfmarathon_model.pkl')
model = model_package['model']

X = pd.DataFrame({
    'age': [30],
    'gender_encoded': [1],  # M=1, K=0
    'time_5km_seconds': [1350]  # 22:30
})

prediction = model.predict(X)[0]
print(f"Predicted time: {prediction/60:.2f} minutes")
```

## 📊 Monitoring z Langfuse

1. **Zarejestruj się na Langfuse**: https://cloud.langfuse.com
2. **Utwórz nowy projekt**
3. **Skopiuj API keys** do `.env`
4. **Sprawdź metryki**:
   - Liczba zapytań
   - Czas odpowiedzi
   - Koszty API
   - Jakość ekstrakcji

## 🔧 Konfiguracja modelu

Domyślne parametry w `config.py`:

```python
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "learning_rate": 0.1,
    "random_state": 42,
}
```

Możesz dostosować te parametry przed treningiem modelu.

## 📈 Metryki modelu

Model osiąga następujące metryki (przykładowe):
- **MAE**: ~3-5 minut
- **RMSE**: ~5-7 minut  
- **R²**: ~0.85-0.90

## 🎨 Przykłady użycia

### Przykład 1: Podstawowe dane
**Input**: "Jestem 30-letnim mężczyzną, 5km biegnę w 22:30"  
**Output**: 01:38:45

### Przykład 2: Niejasny format
**Input**: "Kobieta, 25 lat, pięciokę robię w około 25 minut"  
**Output**: 01:52:30

### Przykład 3: Minimalny opis
**Input**: "45 lat, facet, 20 min na 5k"  
**Output**: 01:28:15

## 🔒 Bezpieczeństwo

- ✅ Nigdy nie commituj pliku `.env`
- ✅ Używaj zmiennych środowiskowych w production
- ✅ Regularnie rotuj API keys
- ✅ Ogranicz dostęp do Spaces (ACL)
- ✅ Używaj HTTPS w production

## 🐛 Rozwiązywanie problemów

### Problem: Model się nie ładuje

```bash
# Sprawdź czy model istnieje w Spaces
python -c "from utils.spaces_handler import SpacesHandler; print(SpacesHandler().list_files('models/'))"

# Pobierz model ręcznie
python -c "from utils.spaces_handler import download_model; download_model('latest_halfmarathon_model.pkl', 'models/halfmarathon_model.pkl')"
```

### Problem: Błąd połączenia z Spaces

```bash
# Sprawdź credentials
python -c "import boto3; print(boto3.client('s3', endpoint_url='https://fra1.digitaloceanspaces.com').list_buckets())"
```

### Problem: LLM nie działa

```bash
# Test OpenAI API
python -c "from openai import OpenAI; client = OpenAI(); print(client.models.list())"
```

## 📝 Roadmap

- [ ] Dodanie więcej cech (waga, BMI, historia treningów)
- [ ] Model ensemble
- [ ] Progressive Web App (PWA)
- [ ] Eksport prognoz do PDF
- [ ] Porównanie z innymi biegaczami
- [ ] Plany treningowe

## 🤝 Contributing

1. Fork projektu
2. Utwórz branch (`git checkout -b feature/AmazingFeature`)
3. Commit zmian (`git commit -m 'Add some AmazingFeature'`)
4. Push do brancha (`git push origin feature/AmazingFeature`)
5. Otwórz Pull Request

## 📄 Licencja

MIT License - zobacz plik LICENSE

## 👥 Autorzy

- Twoje Imię - [GitHub](https://github.com/yourusername)

## 🙏 Podziękowania

- Dane: Półmaraton Wrocław 2023-2024
- Stack: Streamlit, scikit-learn, XGBoost, OpenAI, Langfuse
- Hosting: Digital Ocean

---

**Pytania?** Otwórz issue na GitHubie!
