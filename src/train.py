import os
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, roc_auc_score)
from sklearn.preprocessing import LabelEncoder

BASE_DIR    = Path(__file__).parent.parent
DATA_DIR    = BASE_DIR / "Data"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR  = BASE_DIR / "models"

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

def save(fname):
    path = RESULTS_DIR / fname
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Graf sacuvan: results/{fname}")

# ============================================================
# 1. UCITAVANJE PODATAKA
# ============================================================
df = pd.read_csv(DATA_DIR / "BMW_Car_Sales_Classification.csv")

print("=" * 60)
print("OSNOVNE INFORMACIJE O DATASETU")
print("=" * 60)
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

for col in df.select_dtypes(include='number').columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Nedostajuce vrednosti su popunjene (numericke -> median, kategorijske -> mod).")
print()

# ============================================================
# 3. RASPODELA CILJNE PROMENLJIVE
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
    print("Klase su ravnomerno zastupljene — nema potrebe za balansiranjem.")
else:
    print("Klase NISU ravnomerno zastupljene — treba razmotriti balansiranje (npr. SMOTE).")
print()

plt.figure(figsize=(6, 4))
vc.plot(kind='bar', color=['steelblue', 'salmon'], edgecolor='black')
plt.title("Raspodela klasa - Sales_Classification")
plt.xlabel("Klasa")
plt.ylabel("Broj primeraka")
plt.xticks(rotation=0)
save("01_raspodela_klasa.png")

# ============================================================
# 4. ANALIZA I ODLUKA O Sales_Volume
# ============================================================
print("=" * 60)
print("ANALIZA Sales_Volume")
print("=" * 60)
print(df.groupby('Sales_Classification')['Sales_Volume'].describe())
print()

plt.figure(figsize=(7, 5))
df.boxplot(column='Sales_Volume', by='Sales_Classification', grid=False,
           boxprops=dict(color='steelblue'),
           medianprops=dict(color='red'))
plt.title("Sales_Volume po klasifikaciji prodaje")
plt.suptitle("")
plt.xlabel("Sales_Classification")
plt.ylabel("Sales_Volume")
save("00_sales_volume_analiza.png")

print("ODLUKA: Sales_Volume se izbacuje iz modela.")
print("Razlog: Graf i statistike pokazuju direktnu vezu izmedju Sales_Volume")
print("i Sales_Classification — klasifikacija je verovatno kreirana na osnovu")
print("volumena prodaje. Koriscenje bi predstavljalo data leakage, jer model")
print("ne bi ucio stvarne obrasce, vec bi 'varao' koristeci izvedenu kolonu.")
print()

df.drop(columns=['Sales_Volume'], inplace=True)

# ============================================================
# 5. KORELACIONA ANALIZA NUMERICKIH ATRIBUTA
# ============================================================
print("=" * 60)
print("KORELACIONA ANALIZA NUMERICKIH ATRIBUTA")
print("=" * 60)
numericke_heatmap = ['Year', 'Engine_Size_L', 'Mileage_KM', 'Price_USD']
korelacije = df[numericke_heatmap].corr()
print(korelacije)
print()

plt.figure(figsize=(7, 5))
sns.heatmap(korelacije, annot=True, fmt=".2f", cmap='coolwarm', square=True,
            linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Korelaciona matrica numerickih atributa")
save("01b_korelaciona_matrica.png")

# ============================================================
# 6. GRAFICKI PRIKAZI ODNOSA ATRIBUTA I CILJNE PROMENLJIVE
# ============================================================
kategorijske = ['Model', 'Region', 'Color', 'Fuel_Type', 'Transmission']
filenames    = ['02_model', '03_region', '04_color', '05_fuel_type', '06_transmission']

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

numericke = ['Year', 'Engine_Size_L', 'Mileage_KM', 'Price_USD']
num_files = ['07_year', '08_engine', '09_mileage', '10_price']

for col, fname in zip(numericke, num_files):
    plt.figure(figsize=(7, 5))
    df.boxplot(column=col, by='Sales_Classification', grid=False,
               boxprops=dict(color='steelblue'),
               medianprops=dict(color='red'))
    plt.title(f"{col} po klasifikaciji prodaje")
    plt.suptitle("")
    plt.xlabel("Sales_Classification")
    plt.ylabel(col)
    save(f"{fname}.png")

print()

# ============================================================
# 7. ENCODING KATEGORIJSKIH KOLONA
# ============================================================
print("=" * 60)
print("ENCODING KATEGORIJSKIH KOLONA (Label Encoding)")
print("=" * 60)
print("Odabrana metoda: Label Encoding")
print("Razlog: Random Forest algoritam ne zahteva one-hot encoding —")
print("radi ispravno sa celobrojnim vrednostima koje dodeljuje Label Encoding.")
print("One-hot encoding bi eksponencijalno povecao broj kolona (posebno za")
print("atribute poput Model i Color sa mnogo kategorija), sto bi usporilo")
print("treniranje bez poboljsanja performansi modela.")
print()

le_dict = {}
for col in kategorijske:
    le_dict[col] = LabelEncoder()
    df[col] = le_dict[col].fit_transform(df[col])
    print(f"  {col} -> enkodovano")

le_target = LabelEncoder()
df['Sales_Classification'] = le_target.fit_transform(df['Sales_Classification'])
print("  Sales_Classification -> enkodovano")
print()

# ============================================================
# 8. PRIPREMA PODATAKA ZA MODEL
# ============================================================
X = df.drop(columns=['Sales_Classification'])
y = df['Sales_Classification']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Trening skup: {X_train.shape[0]} redova")
print(f"Test skup:    {X_test.shape[0]} redova")
print()

# ============================================================
# 9. IZBOR I OBRAZLOZENJE MODELA
# ============================================================
print("=" * 60)
print("IZBOR MODELA")
print("=" * 60)
print("Odabrani algoritam: Random Forest (ansambl metoda - Bagging)")
print()
print("Razmatrani alternativni modeli:")
print("  - Decision Tree: interpretabilan, ali nestabilan i sklon overfittingu")
print("  - SVM: dobar za visoku dimenzionalnost, ali spor na vecim skupovima i")
print("    zahteva skaliranje atributa (sto nije neophodno kod stabala odlucivanja)")
print("  - Gradient Boosting: visoka tacnost, ali osetljiv na hiperparametre")
print()
print("Zasto Random Forest?")
print("  - Kao ansambl Bagging metoda, kombinuje vise stabala odlucivanja")
print("    i smanjuje varijansu pojedinacnog stabla.")
print("  - Ne zahteva skaliranje atributa (za razliku od KNN, SVM i sl.) —")
print("    stabla odlucivanja dele podatke po vrednostima, ne po rastojanju.")
print("  - Otporan je na overfitting i sum u podacima.")
print("  - Pruza feature importance -- uvid u koji atributi najvise uticu.")
print("  - Podeseni hiperparametri: n_estimators=100 (broj stabala),")
print("    random_state=42 (reproduktivnost), n_jobs=-1 (paralelno treniranje)")
print()

# ============================================================
# 10. TRENIRANJE MODELA
# ============================================================
print("=" * 60)
print("TRENIRANJE MODELA - Random Forest")
print("=" * 60)

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("Model istreniran.")
print()

# ============================================================
# 11. EVALUACIJA MODELA
# ============================================================
print("=" * 60)
print("EVALUACIJA MODELA")
print("=" * 60)

y_pred       = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

print("--- Klasifikacioni izvestaj ---")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

n_classes = len(le_target.classes_)
if n_classes == 2:
    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
else:
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
print(f"ROC-AUC score: {roc_auc:.4f}")
print("  (ROC-AUC meri sposobnost modela da razlikuje klase;")
print("   vrednost 1.0 = savrsena separacija, 0.5 = nasumicno pogadjanje)")
print()

print("--- K-fold unakrsna validacija (k=5) ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
print(f"Tacnosti po foldovima: {[round(s, 4) for s in cv_scores]}")
print(f"Prosecna tacnost (CV): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
print("  (Unakrsna validacija proverava da li model dobro generalizuje —")
print("   mala standardna devijacija ukazuje na stabilan model.)")
print()

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_target.classes_)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=False, cmap='Blues')
plt.title("Confusion Matrix")
save("11_confusion_matrix.png")

# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================
print("=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)
print(importances)
print()

plt.figure(figsize=(9, 5))
importances.plot(kind='bar', color='steelblue', edgecolor='black')
plt.title("Feature Importance - uticaj atributa na predikciju")
plt.xlabel("Atribut")
plt.ylabel("Vaznost")
plt.xticks(rotation=45, ha='right')
save("12_feature_importance.png")

# ============================================================
# 13. CUVANJE MODELA I ENKODERA
# ============================================================
print("=" * 60)
print("CUVANJE MODELA")
print("=" * 60)

joblib.dump(model,     MODELS_DIR / "model.pkl")
joblib.dump(le_dict,   MODELS_DIR / "le_dict.pkl")
joblib.dump(le_target, MODELS_DIR / "le_target.pkl")
joblib.dump(list(X.columns), MODELS_DIR / "feature_columns.pkl")

print("Sacuvano:")
print("  models/model.pkl          — istrenirani Random Forest model")
print("  models/le_dict.pkl        — enkoderi za kategorijske atribute")
print("  models/le_target.pkl      — enkoder za ciljnu promenljivu")
print("  models/feature_columns.pkl — redosled kolona za predikciju")
print()

# ============================================================
# 14. ZAKLJUCAK
# ============================================================
print("=" * 60)
print("ZAKLJUCAK")
print("=" * 60)
top3     = importances.head(3).index.tolist()
accuracy = accuracy_score(y_test, y_pred)

print(f"Tacnost modela na test skupu: {accuracy * 100:.2f}%")
print(f"ROC-AUC score:                {roc_auc:.4f}")
print(f"Prosecna CV tacnost (k=5):    {cv_scores.mean() * 100:.2f}% +/- {cv_scores.std() * 100:.2f}%")
print()

print("--- Obrazlozenje izbora metrike ---")
print("  Odabrane metrike: Accuracy, Precision, Recall, F1-score, ROC-AUC")
print("  Razlog: Radi se o binarnoj klasifikaciji (High/Low prodaja).")
print("  - Accuracy daje opstu sliku tacnosti modela.")
print("  - Precision i Recall su vazni jer greska u obe klase ima poslovni znacaj")
print("    (lazno pozitivni — nepotrebno angazovani resursi;")
print("     lazno negativni — propustena prodajna sansa).")
print("  - F1-score balansira Precision i Recall u jednu meru.")
print("  - ROC-AUC meri sposobnost modela da razdvoji klase bez obzira na prag.")
print()

print("--- Da li rezultat zadovoljava postavljeni cilj? ---")
if accuracy >= 0.80:
    print(f"  DA — Tacnost od {accuracy * 100:.2f}% prelazi prag od 80%,")
    print("  sto se smatra prihvatljivim za poslovne aplikacije ovog tipa.")
elif accuracy >= 0.70:
    print(f"  DELIMICNO — Tacnost od {accuracy * 100:.2f}% je zadovoljavajuca,")
    print("  ali postoji prostor za poboljsanje (cilj: >80%).")
else:
    print(f"  NE — Tacnost od {accuracy * 100:.2f}% nije zadovoljavajuca.")
    print("  Potrebne su dodatne izmene modela ili skupa podataka.")
if roc_auc >= 0.85:
    print(f"  ROC-AUC od {roc_auc:.4f} ukazuje na odlicnu sposobnost separacije klasa.")
elif roc_auc >= 0.70:
    print(f"  ROC-AUC od {roc_auc:.4f} ukazuje na dobru sposobnost separacije klasa.")
else:
    print(f"  ROC-AUC od {roc_auc:.4f} ukazuje na slabu sposobnost separacije klasa.")
print()

print(f"Najuticajniji atributi su: {', '.join(top3)}")
print("Tumacenje:")
for attr in top3:
    opisi = {
        'Price_USD':     "  - Price_USD: Cena automobila ima najveci uticaj — skuplja vozila se prodaju drugacije od jeftinijih.",
        'Mileage_KM':    "  - Mileage_KM: Kilometraza direktno utice na atraktivnost vozila — niza kilometraza povecava sansu uspesne prodaje.",
        'Year':          "  - Year: Godina proizvodnje odredjuje starost vozila sto je kljucni faktor pri odluci o kupovini.",
        'Engine_Size_L': "  - Engine_Size_L: Zapremina motora utice na segment trzista i tip kupca.",
        'Model':         "  - Model: Odredjeni modeli BMW-a imaju vecu traznju na trzistu.",
        'Region':        "  - Region: Trzisni uslovi i preferencije kupaca se razlikuju po regionima.",
        'Color':         "  - Color: Boja vozila moze uticati na brzinu prodaje.",
        'Fuel_Type':     "  - Fuel_Type: Tip goriva sve vise utice na odluku kupca (elektricni vs. benzinski).",
        'Transmission':  "  - Transmission: Tip menjaca je vazan faktor u zavisnosti od trzista.",
    }
    print(opisi.get(attr, f"  - {attr}: Znacajan prediktor uspesnosti prodaje."))
print()
print("Prakticna upotrebljivost:")
print("  Model moze biti koriscen kao podrska pri poslovnim odlukama —")
print("  npr. za procenu da li ce odredjeni model automobila sa datim")
print("  karakteristikama biti uspesno prodat.")
print()
print("Sales_Volume je izbacen zbog direktne veze sa ciljnom promenljivom (data leakage).")
print()
print("Preporuke za poboljsanje:")
print("  - Isprobati Gradient Boosting (XGBoost) kao alternativu Random Forest-u")
print("  - Uraditi hyperparameter tuning (GridSearchCV ili RandomSearchCV)")
print("    za parametre: n_estimators, max_depth, max_features")
print("  - Prikupiti vise podataka za slabije zastupljene klase")
print("  - Razmotriti SMOTE ako klase nisu ravnomerno zastupljene")
print()
print("Treniranje zavrseno. Sve slike su sacuvane u results/, model u models/")
