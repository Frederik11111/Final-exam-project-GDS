from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
import pandas as pd

# Baseline model: Logistic Regression på stemmed tekst (kun ord)
print("1. Indlæser urenset og splittet data...")
train_data = pd.read_csv('train.csv')
val_data = pd.read_csv('val.csv')

# Håndter tomme felter
train_data['stemmed'] = train_data['stemmed'].fillna('')
val_data['stemmed'] = val_data['stemmed'].fillna('')


print("2. Klargør teksten...")
# Hjælperfunktion til at fjerne de firkantede parenteser og kommaer fra stemmed-kolonnen
train_texts = train_data['stemmed'].str.replace(r"[\[\]',]", "", regex=True)
val_texts = val_data['stemmed'].str.replace(r"[\[\]',]", "", regex=True)

# Labels
y_train = train_data['label']
y_val = val_data['label']

print("3. Vektoriserer teksten (Tager de 10.000 mest brugte ord)...")
# Bruger CountVectorizer som en simpel baseline for at konvertere tekst til numeriske features
vectorizer = CountVectorizer(max_features=10000)

# Fit på træningsdata og transformér både træning og valideringstekst
X_train = vectorizer.fit_transform(train_texts)
X_val = vectorizer.transform(val_texts)

print("4. Træner Logistic Regression modellen (Baseline)...")
# Vi bruger en simpel Logistic Regression som baseline model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)


print("5. Evaluerer modellen på valideringsdata...")
# Forudsig på valideringsdataet
y_pred = model.predict(X_val)

# Print classification report
print("\n", classification_report(y_val, y_pred, target_names=['Reliable News', 'Fake News']))