"""
Half Marathon Predictor — Training Pipeline
============================================
Wczytuje dane z Digital Ocean Spaces (lub lokalnie),
trenuje model GradientBoosting i zapisuje go lokalnie + w Spaces.

Użycie:
    python train_quick.py
    python train_quick.py --local   # pomiń Spaces, użyj lokalnych CSV
"""
import sys
import argparse
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')  # bez GUI
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# UTILS
# ============================================================
def time_to_seconds(t):
    """Konwertuje HH:MM:SS lub MM:SS na sekundy."""
    if pd.isna(t):
        return np.nan
    try:
        parts = str(t).strip().split(':')
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
    except Exception:
        return np.nan


def cat_age_midpoint(cat):
    """Wyciąga środek przedziału wiekowego z kategorii, np. 'M30' -> 35."""
    if pd.isna(cat):
        return np.nan
    digits = ''.join(filter(str.isdigit, str(cat)))
    return int(digits) + 5 if digits else np.nan


def seconds_to_hhmmss(s):
    """Konwertuje sekundy na format HH:MM:SS."""
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


FEATURES = ['age', 'gender_enc', 'time_5km_s']


# ============================================================
# STEP 1 — LOAD DATA
# ============================================================
def load_data(use_spaces: bool, data_dir: Path) -> pd.DataFrame:
    filenames = {
        "halfmarathon_wroclaw_2024.csv": 2024,
        "halfmarathon_wroclaw_2023.csv": 2023,
    }

    if use_spaces:
        print("Pobieranie danych z Digital Ocean Spaces...")
        try:
            import config
            from utils.spaces_handler import download_data_file
            data_dir.mkdir(exist_ok=True)
            for fname in filenames:
                download_data_file(fname, str(data_dir / fname))
        except Exception as e:
            print(f"  Blad Spaces: {e}. Przelaczam na lokalny tryb.")
            use_spaces = False

    frames = []
    upload_dir = Path('/mnt/user-data/uploads')
    for fname, year in filenames.items():
        local = data_dir / fname
        if not local.exists():
            # szukaj w uploadach
            for raw in sorted(upload_dir.glob('*halfmarathon*')):
                if str(year) in raw.name:
                    local = raw
                    break
        if local.exists():
            df = pd.read_csv(local, sep=';')
            df['year'] = year
            frames.append(df)
            print(f"  OK {local.name}: {len(df)} rekordow")
        else:
            print(f"  BRAK: {fname}")

    if not frames:
        raise FileNotFoundError("Brak plikow z danymi!")
    return pd.concat(frames, ignore_index=True)


# ============================================================
# STEP 2 — PREPARE
# ============================================================
def prepare_data(df: pd.DataFrame):
    df = df[df['Plec'].isin(['M', 'K'])].copy() if 'Plec' in df.columns else df[df['Płeć'].isin(['M', 'K'])].copy()
    gender_col = 'Płeć' if 'Płeć' in df.columns else 'Plec'

    df['total_time_s'] = df['Czas'].apply(time_to_seconds)
    df['time_5km_s']   = df['5 km Czas'].apply(time_to_seconds)

    birth_col = 'Rocznik'
    df['birth_year'] = df[birth_col]
    df.loc[(df['birth_year'] < 1924) | (df['birth_year'] > 2008), 'birth_year'] = np.nan
    df['age'] = df['year'] - df['birth_year']
    df['age'] = df['age'].fillna(df['Kategoria wiekowa'].apply(cat_age_midpoint))

    le = LabelEncoder()
    le.fit(['K', 'M'])
    df['gender_enc'] = le.transform(df[gender_col])

    df = df.dropna(subset=['total_time_s', 'time_5km_s', 'age'])
    df = df[
        (df['total_time_s'] >= 3600)  & (df['total_time_s'] <= 18000) &
        (df['time_5km_s']   >= 600)   & (df['time_5km_s']   <= 3600)  &
        (df['age']          >= 16)    & (df['age']          <= 85)
    ]

    print(f"\nDane po czyszczeniu: {len(df)} rekordow")
    print(f"  Plec: M={df[gender_col].eq('M').sum()}, K={df[gender_col].eq('K').sum()}")
    print(f"  Wiek: {df['age'].min():.0f}-{df['age'].max():.0f} (sr. {df['age'].mean():.1f})")
    print(f"  5km:  {df['time_5km_s'].min()/60:.1f}-{df['time_5km_s'].max()/60:.1f} min")
    print(f"  Cel:  {df['total_time_s'].min()/60:.0f}-{df['total_time_s'].max()/60:.0f} min")

    return df, le


# ============================================================
# STEP 3 — FEATURE ANALYSIS
# ============================================================
def feature_analysis(df: pd.DataFrame, output_dir: Path):
    corr = df[FEATURES + ['total_time_s']].corr()['total_time_s'].drop('total_time_s')
    print("\nKorelacja cech z czasem ukonczen:")
    for feat, c in corr.items():
        bar = '#' * int(abs(c) * 30)
        print(f"  {feat:20s}: {c:+.4f}  {bar}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Analiza danych — Polmaraton Wroclaw 2023-2024', fontsize=14)

    gender_col = 'Płeć' if 'Płeć' in df.columns else 'Plec'
    colors = {'M': 'steelblue', 'K': 'tomato'}

    axes[0, 0].hist(df['total_time_s'] / 60, bins=60, color='steelblue', edgecolor='white', linewidth=0.3)
    axes[0, 0].set_xlabel('Czas ukonczenia (min)')
    axes[0, 0].set_title('Rozklad czasu ukonczenia')
    axes[0, 0].axvline(df['total_time_s'].mean()/60, color='red', linestyle='--', label=f"sr={df['total_time_s'].mean()/60:.0f}min")
    axes[0, 0].legend()

    for gender, grp in df.groupby(gender_col):
        axes[0, 1].scatter(grp['time_5km_s']/60, grp['total_time_s']/60, alpha=0.12, s=4,
                           color=colors.get(gender, 'gray'), label=gender)
    axes[0, 1].set_xlabel('Czas na 5km (min)')
    axes[0, 1].set_ylabel('Czas calkowity (min)')
    axes[0, 1].set_title(f'Czas 5km vs. Calkowity (r={corr["time_5km_s"]:.3f})')
    axes[0, 1].legend()

    axes[1, 0].scatter(df['age'], df['total_time_s']/60, alpha=0.07, s=3, color='purple')
    axes[1, 0].set_xlabel('Wiek')
    axes[1, 0].set_ylabel('Czas (min)')
    axes[1, 0].set_title(f'Wiek vs. Czas (r={corr["age"]:.3f})')

    df.boxplot(column='total_time_s', by=gender_col, ax=axes[1, 1])
    axes[1, 1].set_ylabel('Czas (s)')
    axes[1, 1].set_title('Czas wg plci')
    plt.sca(axes[1, 1])

    plt.tight_layout()
    out = output_dir / 'data_analysis.png'
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Wykres: {out}")


# ============================================================
# STEP 4 — TRAIN & COMPARE
# ============================================================
def train(df: pd.DataFrame):
    X = df[FEATURES]
    y = df['total_time_s']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    candidates = {
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.08,
            subsample=0.8, min_samples_leaf=10, random_state=42
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=5,
            n_jobs=-1, random_state=42
        ),
        'Ridge': Ridge(alpha=1.0),
    }

    results = {}
    print(f"\n  {'Model':22s}  MAE(min)  RMSE(min)    R2")
    print("  " + "-" * 52)
    for name, m in candidates.items():
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2   = r2_score(y_test, y_pred)
        results[name] = dict(model=m, mae=mae, rmse=rmse, r2=r2, y_pred=y_pred)
        print(f"  {name:22s}  {mae/60:6.2f}    {rmse/60:6.2f}    {r2:.4f}")

    best_name = min(results, key=lambda k: results[k]['mae'])
    print(f"\nNajlepszy model: {best_name}")
    return results[best_name], best_name, y_test


# ============================================================
# STEP 5 — PLOT RESULTS
# ============================================================
def plot_results(best: dict, best_name: str, y_test, output_dir: Path):
    y_pred = best['y_pred']
    errors_min = (y_test - y_pred) / 60

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Model: {best_name} | MAE={best["mae"]/60:.2f}min | R2={best["r2"]:.4f}')

    axes[0].scatter(y_test/60, y_pred/60, alpha=0.15, s=5, color='steelblue')
    lims = [y_test.min()/60, y_test.max()/60]
    axes[0].plot(lims, lims, 'r--', lw=2)
    axes[0].set_xlabel('Rzeczywisty czas (min)')
    axes[0].set_ylabel('Przewidywany czas (min)')
    axes[0].set_title('Rzeczywisty vs. Przewidywany')

    axes[1].hist(errors_min, bins=60, color='coral', edgecolor='white', linewidth=0.3)
    axes[1].axvline(0, color='red', linestyle='--', lw=2)
    axes[1].axvline(errors_min.median(), color='orange', linestyle='--', lw=1.5,
                    label=f'Mediana: {errors_min.median():.2f}min')
    axes[1].set_xlabel('Blad predykcji (min)')
    axes[1].set_title('Rozklad bledow')
    axes[1].legend()

    plt.tight_layout()
    out = output_dir / 'model_results.png'
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Wykres: {out}")


# ============================================================
# STEP 6 — SAVE
# ============================================================
def save_model(best: dict, best_name: str, le: LabelEncoder, models_dir: Path):
    models_dir.mkdir(exist_ok=True)
    pkg = {
        'model': best['model'],
        'label_encoder': le,
        'features': FEATURES,
        'metadata': {
            'model_name': best_name,
            'features': FEATURES,
            'mae_seconds': float(best['mae']),
            'rmse_seconds': float(best['rmse']),
            'r2': float(best['r2']),
            'training_date': datetime.now().isoformat(),
            'gender_mapping': {'K': 0, 'M': 1},
        }
    }
    local_path = models_dir / 'halfmarathon_model.pkl'
    joblib.dump(pkg, local_path)
    print(f"\nModel zapisany: {local_path}")

    try:
        from utils.spaces_handler import upload_model
        upload_model(str(local_path), 'latest_halfmarathon_model.pkl')
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        upload_model(str(local_path), f'v{ts}_halfmarathon_model.pkl')
        print("Model przeslany do Digital Ocean Spaces!")
    except Exception as e:
        print(f"Spaces upload pominiety: {e}")
    return local_path


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', action='store_true')
    args = parser.parse_args()

    BASE       = Path(__file__).parent
    data_dir   = BASE / 'data'
    models_dir = BASE / 'models'

    print("=" * 60)
    print(" HALF MARATHON PREDICTOR — TRAINING PIPELINE")
    print("=" * 60)

    df_raw         = load_data(use_spaces=not args.local, data_dir=data_dir)
    df, le         = prepare_data(df_raw)

    print("\nFeature analysis...")
    feature_analysis(df, models_dir)

    best, best_name, y_test = train(df)

    print("\nGenerowanie wykresow...")
    plot_results(best, best_name, y_test, models_dir)

    local_path = save_model(best, best_name, le, models_dir)

    print("\nDemo predykcji:")
    demo = [
        (30, 'M', '22:30'), (25, 'K', '25:00'),
        (45, 'M', '20:00'), (35, 'K', '28:00'),
        (50, 'M', '30:00'), (22, 'K', '22:00'),
    ]
    pkg = joblib.load(local_path)
    for age, gender, t5km in demo:
        parts = t5km.split(':')
        t5s = int(parts[0]) * 60 + int(parts[1])
        g_enc = pkg['label_encoder'].transform([gender])[0]
        pred_s = pkg['model'].predict([[age, g_enc, t5s]])[0]
        print(f"  {gender}, {age:2d} lat, 5km={t5km}  ->  {seconds_to_hhmmss(pred_s)}")

    print(f"\n{'='*60}")
    print(f" GOTOWE! MAE={best['mae']/60:.2f}min | R2={best['r2']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
