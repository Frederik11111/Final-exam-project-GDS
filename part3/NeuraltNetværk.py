import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

print("1. Indlæser urenset og splittet data...")

train_data = pd.read_csv('train.csv') # Indlæser træningsdata
val_data = pd.read_csv('val.csv') # Indlæser valideringsdata

train_data['stemmed'] = train_data['stemmed'].fillna('') # Håndter manglende værdier i 'stemmed' kolonnen for træningsdata
val_data['stemmed'] = val_data['stemmed'].fillna('')    # Håndter manglende værdier i 'stemmed' kolonnen for valideringsdata

print("2. Klargør teksten...")
train_texts = train_data['stemmed'].str.replace(r"[\[\]',]", "", regex=True) # Hjælperfunktion til at fjerne '[]' og ',' fra stemmed-kolonnen i træningsdata
val_texts = val_data['stemmed'].str.replace(r"[\[\]',]", "", regex=True) # Hjælperfunktion til at fjerne []' og ',' fra stemmed-kolonnen i valideringsdata

y_train = train_data['label']   # Labels for træningsdata (0 = Reliable, 1 = Fake News)
y_val = val_data['label']   

print("3. Vektoriserer teksten med TF-IDF...")
# Vi bruger TF-IDF i stedet for CountVectorizer for mere komplekse features
tfidf_vectorizer = TfidfVectorizer(max_features=10000) # Begræns til de 10.000 mest informative ord
X_train_tfidf = tfidf_vectorizer.fit_transform(train_texts) # Fit på træningsdata og transformér både træning og valideringstekst
X_val_tfidf = tfidf_vectorizer.transform(val_texts)         # Transformér valideringstekst med den samme vektorizer

print("4. Træner Neuralt Netværk - vær tålmodig!")

# Vi bruger en Multi-layer Perceptron som vores neurale netværksmodel
nn_model = MLPClassifier(
    hidden_layer_sizes=(100,),      # En enkelt skjult lag med 100 neuroner
    activation='relu',              # ReLU aktiveringsfunktion
    solver='adam',                  # Adam optimizer
    batch_size=256,                 # Større batch size for hurtigere træning
    early_stopping=True,            # Stop træning tidligt hvis valideringsscore ikke forbedres
    random_state=42,                # For reproducibility
    verbose=True                    # Print træningsprogress    
)

nn_model.fit(X_train_tfidf, y_train)   # Træner modellen på TF-IDF features og labels

print("\n5. Evaluerer Neuralt Netværk på valideringsdata")

y_pred_nn = nn_model.predict(X_val_tfidf) # Forudsig på valideringsdataet

# Print classification report
print("\n", classification_report(y_val, y_pred_nn, target_names=['Reliable News', 'Fake News']))