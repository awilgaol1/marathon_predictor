"""
Upload modelu do Digital Ocean Spaces
======================================
Ten skrypt automatycznie czyta klucze z pliku .env
i uploaduje model do Spaces.
"""
import boto3
from pathlib import Path
import os
from dotenv import load_dotenv

# Wczytaj zmienne z .env
load_dotenv()

# Pobierz dane z .env
ACCESS_KEY = os.getenv('DO_SPACES_KEY')
SECRET_KEY = os.getenv('DO_SPACES_SECRET')
BUCKET_NAME = os.getenv('DO_SPACES_BUCKET', 'gotoitanna')
REGION = os.getenv('DO_SPACES_REGION', 'fra1')
ENDPOINT = os.getenv('DO_SPACES_ENDPOINT', 'https://fra1.digitaloceanspaces.com')

def main():
    print("=" * 60)
    print(" UPLOAD MODELU DO DIGITAL OCEAN SPACES")
    print("=" * 60)
    print()
    
    # Sprawdź czy klucze są w .env
    if not ACCESS_KEY or not SECRET_KEY:
        print("❌ BŁĄD: Brak kluczy w pliku .env!")
        print()
        print("Dodaj do pliku .env:")
        print("  DO_SPACES_KEY=DO00...")
        print("  DO_SPACES_SECRET=eF...")
        print("  DO_SPACES_BUCKET=gotoitanna")
        print()
        return
    
    print(f"📋 Konfiguracja z .env:")
    print(f"   Bucket:   {BUCKET_NAME}")
    print(f"   Region:   {REGION}")
    print(f"   Endpoint: {ENDPOINT}")
    print(f"   Key:      {ACCESS_KEY[:10]}... (ukryty)")
    print()
    
    # Utwórz klienta S3
    try:
        s3 = boto3.client(
            's3',
            region_name=REGION,
            endpoint_url=ENDPOINT,
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_KEY
        )
        print("✅ Połączono z Digital Ocean Spaces")
    except Exception as e:
        print(f"❌ Błąd połączenia: {e}")
        return
    
    # Ścieżka do modelu
    model_path = Path('models/halfmarathon_model.pkl')
    
    if not model_path.exists():
        print(f"❌ Model nie znaleziony: {model_path}")
        print()
        print("Uruchom najpierw:")
        print("  python train_quick.py --local")
        return
    
    size_kb = model_path.stat().st_size / 1024
    print(f"📦 Model lokalnie: {model_path} ({size_kb:.1f} KB)")
    print()
    
    # Upload modelu
    print(f"📤 Uploading do Spaces...")
    print(f"   Path: models/latest_halfmarathon_model.pkl")
    
    try:
        s3.upload_file(
            str(model_path),
            BUCKET_NAME,
            'models/latest_halfmarathon_model.pkl',
            ExtraArgs={'ACL': 'private'}
        )
        
        print("✅ Upload successful!")
        print()
        
    except Exception as e:
        print(f"❌ Błąd uploadu: {e}")
        print()
        print("💡 Sprawdź:")
        print("  - Czy klucze w .env są poprawne")
        print("  - Czy bucket istnieje w Digital Ocean")
        return
    
    # Sprawdź co jest w Spaces
    print("📋 Pliki w Spaces (models/):")
    try:
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix='models/')
        
        if 'Contents' in response:
            for obj in response['Contents']:
                size = obj['Size'] / 1024
                print(f"   ✅ {obj['Key']} ({size:.1f} KB)")
        else:
            print("   ⚠️ Brak plików w models/")
    except Exception as e:
        print(f"   ⚠️ Nie można wylistować: {e}")
    
    print()
    print("=" * 60)
    print(" 🎉 GOTOWE!")
    print("=" * 60)
    print()
    print("Model jest w Digital Ocean Spaces!")
    print()
    print("NASTĘPNE KROKI:")
    print("1. Sprawdź w Digital Ocean Spaces → models/")
    print("2. Dodaj zmienne AWS_* w DO App Platform")
    print("3. Poczekaj 5-10 min na rebuild")
    print("4. Testuj aplikację!")
    print()


if __name__ == "__main__":
    main()