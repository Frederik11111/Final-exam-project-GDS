import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack

print("1. Indlæser data fra CSV...")
train_data = pd.read_csv('train.csv')
val_data = pd.read_csv('val.csv')

# Håndter manglende værdier for de kolonner, vi skal bruge
train_data['stemmed'] = train_data['stemmed'].fillna('')
val_data['stemmed'] = val_data['stemmed'].fillna('')
train_data['title'] = train_data['title'].fillna('')
val_data['title'] = val_data['title'].fillna('')

# Labels, 0 for reliable news og 1 for fake news
y_train = train_data['label']
y_val = val_data['label']

print("2. Klargør kombineret tekst (Titel + Indhold)...")
# Fjerner lister-tegn fra den gemte CSV-streng
train_stemmed_clean = train_data['stemmed'].str.replace(r"[\[\]',]", "", regex=True)
val_stemmed_clean = val_data['stemmed'].str.replace(r"[\[\]',]", "", regex=True)

# Sætter titel og den rensede tekst sammen
train_combined_text = train_data['title'] + " " + train_stemmed_clean
val_combined_text = val_data['title'] + " " + val_stemmed_clean

print("3. Vektoriserer den kombinerede tekst...")

# Vektoriserer den kombinerede tekst (Titel + Indhold) ved hjælp af CountVectorizer
vectorizer = CountVectorizer(max_features=10000)
X_train_text = vectorizer.fit_transform(train_combined_text)
X_val_text = vectorizer.transform(val_combined_text)

print("4. Opretter metadata-features (One-Hot Encoding af 'domain')...")
# OneHotEncoder for at konvertere 'domain' kolonnen til numeriske features
encoder = OneHotEncoder(handle_unknown='ignore')
X_train_domain = encoder.fit_transform(train_data[['domain']].fillna('unknown'))
X_val_domain = encoder.transform(val_data[['domain']].fillna('unknown'))

print("5. Samler tekst-features og metadata-features (Stacking)...")
# hstack limer vores tekst-matrix og vores domæne-matrix sammen side om side
X_train_final = hstack([X_train_text, X_train_domain])
X_val_final = hstack([X_val_text, X_val_domain])

print("6. Træner Logistic Regression modellen med Metadata...")
# Vi bruger en simpel Logistic Regression som model, men nu med både tekst og metadata som input
meta_model = LogisticRegression(max_iter=1000, random_state=42)
meta_model.fit(X_train_final, y_train)

print("7. Evaluerer modellen på valideringsdata...")
# Forudsig på valideringsdataet
y_pred_meta = meta_model.predict(X_val_final)

# Print classification report
print("\n", classification_report(y_val, y_pred_meta, target_names=['Reliable News', 'Fake News']))