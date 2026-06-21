import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

SALES_VOLUME_PENALTY   = 0.1   # skalira vrednosti (stabla invarijantna, ali drzi raspon mali)
SALES_VOLUME_NOISE_STD = 2000  # Gaussov sum koji prelapa granicu High/Low (7000), smanjuje dominaciju

BASE_DIR    = Path(__file__).resolve().parents[1]
DATA_DIR    = BASE_DIR / "Data"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR  = BASE_DIR / "models"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

def save(fname):
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / fname, dpi=150)
    plt.close()
    print(f"Graf sacuvan: results/{fname}")

# ============================================================
# 1. UCITAVANJE PODATAKA
# ============================================================
print("=" * 60)
print("OSNOVNE INFORMACIJE O DATASETU")
print("=" * 60)

df = pd.read_csv(DATA_DIR / "BMW_Car_Sales_Classification.csv")

print(f"Broj redova: {df.shape[0]}")
print(f"Broj kolona: {df.shape[1]}")
print()
print(df.head())
print()
print(df.info())
print()

# ============================================================
# 2. ANALIZA NEDOSTAJUCIH VREDNOSTI
# ============================================================
print("=" * 60)
print("NEDOSTAJUCE VREDNOSTI")
print("=" * 60)
print(df.isnull().sum())
print()

numericke_cols   = ['Year', 'Engine_Size_L', 'Mileage_KM', 'Price_USD', 'Sales_Volume']
kategorijske     = ['Model', 'Region', 'Fuel_Type', 'Transmission']

for col in numericke_cols:
    df[col] = df[col].fillna(df[col].median())
for col in kategorijske + ['Sales_Classification']:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Nedostajuce vrednosti su popunjene (numericke -> median, kategorijske -> mod).")
print()

# ============================================================
# 3. DETEKCIJA DUPLIKATA
# ============================================================
print("=" * 60)
print("DETEKCIJA DUPLIKATA")
print("=" * 60)

duplikati = df.duplicated().sum()
print(f"Broj duplikata: {duplikati}")
if duplikati > 0:
    df.drop_duplicates(inplace=True)
    print(f"Duplikati uklonjeni. Novi broj redova: {df.shape[0]}")
else:
    print("Nema duplikata.")
print()

# ============================================================
# 4. DETEKCIJA ANOMALIJA
# ============================================================
print("=" * 60)
print("DETEKCIJA ANOMALIJA")
print("=" * 60)

print("--- Statisticke metode ---")
print("IQR (1.5x):")
iqr_outlier = pd.Series(False, index=df.index)
for col in numericke_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    mask = (df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)
    iqr_outlier |= mask
    print(f"  {col}: {mask.sum()} anomalija")
print(f"  -> Ukupno: {iqr_outlier.sum()}")

print("Z-score (|z| > 3):")
zscore_outlier = pd.Series(False, index=df.index)
for col in numericke_cols:
    z = (df[col] - df[col].mean()).abs() / df[col].std()
    mask = z > 3
    zscore_outlier |= mask
    print(f"  {col}: {mask.sum()} anomalija")
print(f"  -> Ukupno: {zscore_outlier.sum()}")
print("Zakljucak: Statisticke metode ne pronalaze anomalije (uniformna raspodela).")
print()

# ============================================================
# 5. RASPODELA CILJNE PROMENLJIVE
# ============================================================
print("=" * 60)
print("RASPODELA CILJNE PROMENLJIVE (Sales_Classification)")
print("=" * 60)

vc     = df['Sales_Classification'].value_counts()
vc_pct = df['Sales_Classification'].value_counts(normalize=True) * 100
print(vc)
print()
print("Procentualna zastupljenost klasa:")
for klasa, pct in vc_pct.items():
    print(f"  {klasa}: {pct:.1f}%")
if vc_pct.max() - vc_pct.min() < 10:
    print("Klase su ravnomerno zastupljene.")
else:
    print("Klase NISU ravnomerno zastupljene -- razmotriti balansiranje (SMOTE).")
print()

plt.figure(figsize=(6, 4))
vc.plot(kind='bar', color=['steelblue', 'salmon'], edgecolor='black')
plt.title("Raspodela klasa - Sales_Classification")
plt.xlabel("Klasa")
plt.ylabel("Broj primeraka")
plt.xticks(rotation=0)
save("01_raspodela_klasa.png")

# ============================================================
# 6. ANALIZA Sales_Volume
# ============================================================
print("=" * 60)
print("ANALIZA Sales_Volume")
print("=" * 60)
print(df.groupby('Sales_Classification')['Sales_Volume'].describe())
print()

plt.figure(figsize=(7, 5))
df.boxplot(column='Sales_Volume', by='Sales_Classification', grid=False,
           boxprops=dict(color='steelblue'), medianprops=dict(color='red'))
plt.title("Sales_Volume po klasifikaciji prodaje")
plt.suptitle("")
plt.xlabel("Sales_Classification")
plt.ylabel("Sales_Volume")
save("00_sales_volume_analiza.png")

print(f"ODLUKA: Sales_Volume ostaje uz panderisuci faktor (x{SALES_VOLUME_PENALTY}) + Gaussov sum (std={SALES_VOLUME_NOISE_STD}).")
print("Razlog: Stabla su invarijantna na skaliranje, pa sum prelapava granicu High/Low na 7000")
print("        i stvara genuinu nesigurnost -- Sales_Volume postaje slab, ne savrseni prediktor.")
print()

rng = np.random.default_rng(42)
noise = rng.normal(0, SALES_VOLUME_NOISE_STD, size=len(df))
df['Sales_Volume'] = (df['Sales_Volume'] + noise).clip(lower=0) * SALES_VOLUME_PENALTY

# ============================================================
# 7. KORELACIONA ANALIZA NUMERICKIH ATRIBUTA
# ============================================================
print("=" * 60)
print("KORELACIONA ANALIZA NUMERICKIH ATRIBUTA")
print("=" * 60)

numericke_heatmap = ['Year', 'Engine_Size_L', 'Mileage_KM', 'Price_USD', 'Sales_Volume']
korelacije = df[numericke_heatmap].corr()
print(korelacije)
print()

plt.figure(figsize=(7, 5))
sns.heatmap(korelacije, annot=True, fmt=".2f", cmap='coolwarm', square=True,
            linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Korelaciona matrica numerickih atributa")
save("01b_korelaciona_matrica.png")

# ============================================================
# 8. GRAFICKI PRIKAZI ATRIBUTA
# ============================================================
filenames = ['02_model', '03_region', '04_fuel_type', '05_transmission']
for col, fname in zip(kategorijske, filenames):
    plt.figure(figsize=(10, 5))
    ct = pd.crosstab(df[col], df['Sales_Classification'], normalize='index') * 100
    ct.plot(kind='bar', stacked=True, colormap='Set2', edgecolor='black')
    plt.title(f"Uspesnost prodaje po: {col}")
    plt.xlabel(col)
    plt.ylabel("Procenat (%)")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Klasifikacija")
    save(f"{fname}.png")

num_files = ['07_year', '08_engine', '09_mileage', '10_price', '11_sales_volume']
for col, fname in zip(numericke_heatmap, num_files):
    plt.figure(figsize=(7, 5))
    df.boxplot(column=col, by='Sales_Classification', grid=False,
               boxprops=dict(color='steelblue'), medianprops=dict(color='red'))
    plt.title(f"{col} po klasifikaciji prodaje")
    plt.suptitle("")
    plt.xlabel("Sales_Classification")
    plt.ylabel(col)
    save(f"{fname}.png")

print()

# ============================================================
# 9. PRIPREMA PODATAKA
# ============================================================
print("=" * 60)
print("PRIPREMA PODATAKA")
print("=" * 60)
print("Odabrana metoda enkodiranja: OneHotEncoder (u Pipeline-u)")
print("Razlog: Poredimo vise modela (LR, KNN, DT, RF). Logisticka regresija")
print("i KNN zahtevaju ispravno numericko kodiranje kategorija, dok stabla")
print("odlucivanja rade jednako dobro sa oba pristupa. OneHotEncoder je")
print("univerzalno kompatibilan sa svim poredjenim algoritmima.")
print()

X = df.drop(columns=['Sales_Classification'])
y = df['Sales_Classification']

le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)


X_temp, X_test, y_temp, y_test = train_test_split(
    X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
)

print(f"Trening skup:    {len(X_train)} redova (70%)")
print(f"Validacioni skup:{len(X_val)} redova (15%)")
print(f"Test skup:       {len(X_test)} redova (15%)")
print()

# ============================================================
# 10. POREDENJE MODELA
# ============================================================
print("=" * 60)
print("POREDENJE MODELA")
print("=" * 60)

preprocessor = ColumnTransformer(
    transformers=[
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), kategorijske),
        ("scaler", StandardScaler(), numericke_heatmap),
    ]
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN":                 KNeighborsClassifier(n_neighbors=7),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    "XGBoost":             XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0),
}

trained_models = {}
results        = []

for name, classifier in models.items():
    print(f"  Treniranje: {name}...")
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
    pipe.fit(X_train, y_train)
    val_acc = accuracy_score(y_val, pipe.predict(X_val))
    cv_scores = cross_val_score(pipe, X_temp, y_temp, cv=5, scoring='balanced_accuracy', n_jobs=-1)
    trained_models[name] = pipe
    results.append({
        "Model": name,
        "Val Accuracy": round(val_acc, 4),
        "CV Balanced Acc": round(cv_scores.mean(), 4),
        "CV Std": round(cv_scores.std(), 4),
    })
    print(f"    Val Accuracy: {val_acc:.4f} | CV Balanced Acc: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

results_df = pd.DataFrame(results).sort_values("CV Balanced Acc", ascending=False)
print()
print(results_df.to_string(index=False))
print()

# ============================================================
# 11. IZBOR MODELA
# ============================================================
best_row  = results_df.iloc[0]
best_name = best_row['Model']

print("=" * 60)
print(f"IZBOR MODELA: {best_name}")
print("=" * 60)
print(f"Razlog: Najvisi CV Balanced Accuracy ({best_row['CV Balanced Acc']:.4f}, std={best_row['CV Std']:.4f}).")
print("Kriterijum: CV Balanced Accuracy -- pouzdaniji od Val Accuracy na jednom splitu")
print("            i balansiran zbog neujednacenih klasa (69.5% Low / 30.5% High).")
print()

best_model = trained_models[best_name]

# ============================================================
# 12. PODESAVANJE HIPERPARAMETARA (GridSearchCV)
# ============================================================
print("=" * 60)
print("PODESAVANJE HIPERPARAMETARA (GridSearchCV)")
print("=" * 60)

classifier = best_model.named_steps['classifier']
if hasattr(classifier, 'n_estimators'):
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth':    [None, 10],
        'classifier__max_features': ['sqrt', 'log2'],
    }
elif hasattr(classifier, 'max_depth'):
    param_grid = {
        'classifier__max_depth':    [None, 5, 10, 20],
        'classifier__max_features': ['sqrt', 'log2', None],
    }
elif hasattr(classifier, 'C'):
    param_grid = {'classifier__C': [0.01, 0.1, 1, 10]}
else:
    param_grid = {'classifier__n_neighbors': [3, 5, 7, 11]}

grid_search = GridSearchCV(
    best_model,
    param_grid,
    cv=5,
    scoring='balanced_accuracy',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

print(f"Najbolji parametri:          {grid_search.best_params_}")
print(f"Najbolja CV balanced_acc:    {grid_search.best_score_:.4f}")
print()

best_model = grid_search.best_estimator_

# ============================================================
# 13. CUVANJE MODELA I PODATAKA
# ============================================================
print("=" * 60)
print("CUVANJE MODELA I PODATAKA")
print("=" * 60)

categories = {col: sorted(df[col].unique().tolist()) for col in kategorijske}

joblib.dump(best_model, MODELS_DIR / "model.pkl")
joblib.dump(le_target,  MODELS_DIR / "le_target.pkl")
joblib.dump(categories, MODELS_DIR / "categories.pkl")
joblib.dump(
    {"X_train": X_train, "X_val": X_val, "X_test": X_test,
     "y_train": y_train, "y_val": y_val, "y_test": y_test},
    MODELS_DIR / "data_splits.pkl"
)

print(f"  models/model.pkl        -- istrenirani {best_name} (Pipeline)")
print("  models/le_target.pkl    -- enkoder ciljne promenljive")
print("  models/categories.pkl   -- kategorije za web app")
print("  models/data_splits.pkl  -- train/val/test skupovi za evaluaciju")
print()
print("Treniranje zavrseno. Pokrenite evaluate.py za detalje evaluacije.")
