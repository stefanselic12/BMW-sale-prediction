import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay, roc_auc_score, accuracy_score)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

BASE_DIR    = Path(__file__).resolve().parents[1]
MODELS_DIR  = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def save(fname):
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / fname, dpi=150)
    plt.close()
    print(f"Graf sacuvan: results/{fname}")

# ============================================================
# UCITAVANJE MODELA I PODATAKA
# ============================================================
model     = joblib.load(MODELS_DIR / "model.pkl")
le_target = joblib.load(MODELS_DIR / "le_target.pkl")
splits    = joblib.load(MODELS_DIR / "data_splits.pkl")

X_train = splits["X_train"]
X_val   = splits["X_val"]
X_test  = splits["X_test"]
y_train = splits["y_train"]
y_val   = splits["y_val"]
y_test  = splits["y_test"]

# ============================================================
# 1. EVALUACIJA NA VALIDACIONOM SKUPU
# ============================================================
print("=" * 60)
print("EVALUACIJA - VALIDACIONI SKUP")
print("=" * 60)

y_val_pred = model.predict(X_val)
print(classification_report(y_val, y_val_pred, target_names=le_target.classes_))

# ============================================================
# 2. EVALUACIJA NA TEST SKUPU (finalna)
# ============================================================
print("=" * 60)
print("EVALUACIJA - TEST SKUP (finalna)")
print("=" * 60)

y_pred       = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

print(classification_report(y_test, y_pred, target_names=le_target.classes_))

n_classes = len(le_target.classes_)
if n_classes == 2:
    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
else:
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')

print(f"ROC-AUC score: {roc_auc:.4f}")
print("  (1.0 = savrsena separacija, 0.5 = nasumicno pogadjanje)")
print()

# ============================================================
# 3. K-FOLD UNAKRSNA VALIDACIJA (na trening skupu)
# ============================================================
print("=" * 60)
print("K-FOLD UNAKRSNA VALIDACIJA (k=5, trening skup)")
print("=" * 60)

cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='balanced_accuracy')

print(f"Balanced Acc po foldovima: {[round(s, 4) for s in cv_scores]}")
print(f"Prosecna balanced acc (CV): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
print("  (Mala std devijacija ukazuje na stabilan model.)")
print()

# ============================================================
# 4. CONFUSION MATRIX
# ============================================================
cm   = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_target.classes_)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=False, cmap='Blues')
plt.title("Confusion Matrix - Test skup")
save("11_confusion_matrix.png")

# ============================================================
# 5. FEATURE IMPORTANCE
# ============================================================
print("=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

classifier    = model.named_steps['classifier']
preprocessor  = model.named_steps['preprocessor']

importances_arr = classifier.feature_importances_

raw_names     = preprocessor.get_feature_names_out()
clean_names   = [n.replace('ohe__', '').replace('scaler__', '').replace('remainder__', '') for n in raw_names]
importances   = pd.Series(importances_arr, index=clean_names)
importances   = importances.sort_values(ascending=False)

print(importances.head(15).to_string())
print()

plt.figure(figsize=(12, 6))
importances.head(15).plot(kind='bar', color='steelblue', edgecolor='black')
plt.title("Feature Importance - top 15 atributa")
plt.xlabel("Atribut")
plt.ylabel("Vaznost")
plt.xticks(rotation=45, ha='right')
save("12_feature_importance.png")

# ============================================================
# 6. POREDENJE: SVI ATRIBUTI vs. TOP ATRIBUTI
# ============================================================
print("=" * 60)
print("POREDENJE: SVI ATRIBUTI vs. TOP ATRIBUTI")
print("=" * 60)

kategorijske  = ['Model', 'Region', 'Fuel_Type', 'Transmission']
numeric_cols  = ['Year', 'Engine_Size_L', 'Mileage_KM', 'Price_USD', 'Sales_Volume']

ohe_transf    = model.named_steps['preprocessor'].named_transformers_['ohe']
ohe_feat_names = ohe_transf.get_feature_names_out(kategorijske)
n_ohe         = len(ohe_feat_names)


col_importance = {col: 0.0 for col in kategorijske + numeric_cols}
for i, feat_name in enumerate(ohe_feat_names):
    for col in kategorijske:
        if feat_name.startswith(col + '_'):
            col_importance[col] += importances_arr[i]
            break
for i, col in enumerate(numeric_cols):
    col_importance[col] = importances_arr[n_ohe + i]

col_imp = pd.Series(col_importance).sort_values(ascending=False)
print("Vaznost po originalnim atributima:")
print(col_imp.to_string())
print()

top_cols = col_imp.head(5).index.tolist()
print(f"Top 5 atributa: {top_cols}")
print()

top_kat = [c for c in top_cols if c in kategorijske]
top_num = [c for c in top_cols if c not in kategorijske]

transformers = []
if top_kat:
    transformers.append(("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), top_kat))
if top_num:
    transformers.append(("scaler", StandardScaler(), top_num))
new_preprocessor = ColumnTransformer(transformers=transformers)

new_pipe = Pipeline([
    ("preprocessor", new_preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])
new_pipe.fit(X_train[top_cols], y_train)

acc_all = accuracy_score(y_test, y_pred)
acc_top = accuracy_score(y_test, new_pipe.predict(X_test[top_cols]))

print(f"Tacnost sa SVIM atributima (9):  {acc_all * 100:.2f}%")
print(f"Tacnost sa TOP 5 atributima:     {acc_top * 100:.2f}%")
print()
if acc_top >= acc_all - 0.02:
    print("Zakljucak: Top 5 atributa daje uporedivu tacnost -- model je efikasan.")
else:
    print("Zakljucak: Koriscenje svih atributa daje znacajno bolju tacnost.")
print()

# ============================================================
# 7. ZAKLJUCAK
# ============================================================
accuracy = accuracy_score(y_test, y_pred)

print("=" * 60)
print("ZAKLJUCAK")
print("=" * 60)
print(f"Tacnost modela na test skupu:       {accuracy * 100:.2f}%")
print(f"ROC-AUC score:                      {roc_auc:.4f}")
print(f"Prosecna CV balanced acc (k=5):     {cv_scores.mean() * 100:.2f}% +/- {cv_scores.std() * 100:.2f}%")
print()

print("--- Obrazlozenje izbora metrike ---")
print("  Odabrane metrike: Accuracy, Precision, Recall, F1-score, ROC-AUC")
print("  - Accuracy: opsta slika tacnosti modela.")
print("  - Precision/Recall: vazni jer obe greske imaju poslovni znacaj")
print("    (lazno pozitivni = nepotrebno angazovani resursi;")
print("     lazno negativni = propustena prodajna sansa).")
print("  - F1-score: balans izmedju Precision i Recall.")
print("  - ROC-AUC: sposobnost razdvajanja klasa bez obzira na prag.")
print()

print("--- Da li rezultat zadovoljava postavljeni cilj? ---")
if accuracy >= 0.80:
    print(f"  DA -- Tacnost od {accuracy * 100:.2f}% prelazi prag od 80%.")
elif accuracy >= 0.70:
    print(f"  DELIMICNO -- Tacnost od {accuracy * 100:.2f}% je zadovoljavajuca, cilj je >80%.")
else:
    print(f"  NE -- Tacnost od {accuracy * 100:.2f}% nije zadovoljavajuca.")
    print("  Potrebne su dodatne izmene modela ili skupa podataka.")

if roc_auc >= 0.85:
    print(f"  ROC-AUC od {roc_auc:.4f} -- odlicna sposobnost separacije klasa.")
elif roc_auc >= 0.70:
    print(f"  ROC-AUC od {roc_auc:.4f} -- dobra sposobnost separacije klasa.")
else:
    print(f"  ROC-AUC od {roc_auc:.4f} -- slaba sposobnost separacije klasa.")
print()

print("--- Napomena o Sales_Volume ---")
print("  Sales_Classification je kreirana direktno iz Sales_Volume (High ako > 7000).")
print("  Sales_Volume je zadrzana uz panderisuci faktor i Gaussov sum (std=2000)")
print("  koji prelapa granicu klasa i smanjuje dominaciju kolone u modelu.")
print()

print("--- Preporuke za poboljsanje ---")
print("  - Prikupiti atribute koji su stvarno uzrocno povezani sa prodajom")
print("    (trendovi trzista, sezonalnost, konkurentska cena, marketing...)")
print("  - Razmotriti SMOTE za balansiranje klasa (69.5% Low vs 30.5% High)")
