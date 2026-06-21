# BMW Car Sales Classification

Projekat mašinskog učenja za klasifikaciju uspešnosti prodaje BMW automobila (High/Low).

## Struktura projekta

```
sausau_pro/
├── Data/
│   └── BMW_Car_Sales_Classification.csv
├── models/
│   ├── model.pkl
│   ├── le_target.pkl
│   ├── categories.pkl
│   └── data_splits.pkl
├── results/
│   └── (grafovi)
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── app/
│   ├── ui.py
│   └── api.py
├── requirements.txt
└── Dockerfile
```

## Instalacija

```bash
pip install -r requirements.txt
```

## Pokretanje

### 1. Treniranje modela
```bash
python src/train.py
```

### 2. Evaluacija modela
```bash
python src/evaluate.py
```

### 3. Test predikcije (skripta)
```bash
python src/predict.py
```

### 4. Web aplikacija (Streamlit UI)
```bash
python -m streamlit run app/ui.py
```
Otvori browser na: `http://localhost:8501`

### 5. REST API (FastAPI)
```bash
uvicorn app.api:app --reload
```
Dokumentacija API-ja: `http://localhost:8000/docs`

## Docker

```bash
docker build -t bmw-sales .
docker run -p 8501:8501 bmw-sales
```

## Rezultati

| Metrika | Vrednost |
|---|---|
| Test Accuracy | 82.76% |
| ROC-AUC | 0.9125 |
| CV Balanced Accuracy | 83.46% |
| Odabrani model | Random Forest |
