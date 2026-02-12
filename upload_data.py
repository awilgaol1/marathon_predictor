"""
Script to upload initial data to Digital Ocean Spaces.
Run this script after setting up your Spaces bucket.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

import config
from utils.spaces_handler import SpacesHandler

def upload_initial_data():
    """Upload CSV files to Digital Ocean Spaces."""
    
    # Files to upload (update paths as needed)
    data_files = [
        {
            'local': '/mnt/user-data/uploads/1770658675244_halfmarathon_wroclaw_2024__final__3_.csv',
            'remote': 'data/halfmarathon_wroclaw_2024.csv'
        },
        {
            'local': '/mnt/user-data/uploads/1770658675249_halfmarathon_wroclaw_2023__final__1_.csv',
            'remote': 'data/halfmarathon_wroclaw_2023.csv'
        }
    ]
    
    spaces = SpacesHandler()
    
    print("🚀 Rozpoczynam upload danych do Digital Ocean Spaces...")
    print(f"Bucket: {config.DO_SPACES_BUCKET}\n")
    
    for file_info in data_files:
        local_path = file_info['local']
        remote_path = file_info['remote']
        
        print(f"📤 Uploading {Path(local_path).name}...")
        
        # Check if file exists locally
        if not Path(local_path).exists():
            print(f"  ❌ Plik lokalny nie istnieje: {local_path}")
            continue
        
        # Upload file
        success = spaces.upload_file(local_path, remote_path)
        
        if success:
            print(f"  ✅ Plik przesłany: {remote_path}")
            # Get public URL
            url = spaces.get_public_url(remote_path)
            print(f"  🔗 URL: {url}")
        else:
            print(f"  ❌ Błąd podczas przesyłania: {remote_path}")
        
        print()
    
    # List all files in the data directory
    print("📋 Pliki w katalogu data/:")
    files = spaces.list_files(prefix="data/")
    for file in files:
        print(f"  - {file}")
    
    print("\n✅ Upload zakończony!")


if __name__ == "__main__":
    upload_initial_data()
