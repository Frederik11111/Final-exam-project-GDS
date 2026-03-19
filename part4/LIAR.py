import pandas as pd
import re
import nltk
from nltk.stem import PorterStemmer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

print("1. Indlæser urenset og splittet data...")
train_data = pd.read_csv('train.csv')   # Indlæser træningsdata
val_data = pd.read_csv('val.csv')       # Indlæser valideringsdata

# Håndter tomme felter
train_data['stemmed'] = train_data['stemmed'].fillna('') # Håndter manglende værdier i 'stemmed' kolonnen for træningsdata
val_data['stemmed'] = val_data['stemmed'].fillna('')     # Håndter manglende værdier i 'stemmed' kolonnen for valideringsdata

print("2. Klargør teksten...")
train_texts = train_data['stemmed'].str.replace(r"[\[\]',]", "", regex=True) # Hjælperfunktion til at fjerne '[]' og ',' fra stemmed-kolonnen i træningsdata
val_texts = val_data['stemmed'].str.replace(r"[\[\]',]", "", regex=True)     # Hjælperfunktion til at fjerne []' og ',' fra stemmed-kolonnen i valideringsdata

y_train = train_data['label']   # Labels for træningsdata (0 = Reliable, 1 = Fake News)
y_val = val_data['label']       # -||-
 

print(" Model 1: Baseline Model (Logistic Regression) ")
print("3. Vektoriserer teksten (Tager de 10.000 mest brugte ord)")
vectorizer = CountVectorizer(max_features=10000)
X_train_count = vectorizer.fit_transform(train_texts)
X_val_count = vectorizer.transform(val_texts)

print("4. Træner Logistic Regression modellen (Baseline)...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_count, y_train)

print("5. Evaluerer modellen på FakeNews valideringsdata...")
y_pred = model.predict(X_val_count)
print("\n", classification_report(y_val, y_pred, target_names=['Reliable News', 'Fake News']))



print(" Model 2: Avanceret Model (Neuralt Netværk) ")

print("Vektoriserer teksten med TF-IDF...")
tfidf_vectorizer = TfidfVectorizer(max_features=10000)      # Begræns til de 10.000 mest informative ord
X_train_tfidf = tfidf_vectorizer.fit_transform(train_texts) # Fit på træningsdata og transformér både træning og valideringstekst
X_val_tfidf = tfidf_vectorizer.transform(val_texts)         # Transformér valideringstekst med den samme vektorizer

print("Træner Neuralt Netværk (MLP)...")
nn_model = MLPClassifier(
    hidden_layer_sizes=(100,),                              
    activation='relu', 
    solver='adam', 
    batch_size=256,
    early_stopping=True, 
    random_state=42,
    verbose=False # Sat til False for et mere rent output
)
nn_model.fit(X_train_tfidf, y_train)

print("Evaluerer Neuralt Netværk på FakeNews valideringsdata...")
y_pred_nn = nn_model.predict(X_val_tfidf)
print("\n", classification_report(y_val, y_pred_nn, target_names=['Reliable News', 'Fake News']))



print(" Cross-domain evaluering PÅ LIAR datasættet")

# Tilpasser LIAR datasættet til vores models inputformat
liar_columns = [
    'id', 'label', 'statement', 'subject', 'speaker', 'job', 
    'state', 'party', 'barely_true_counts', 'false_counts', 
    'half_true_counts', 'mostly_true_counts', 'pants_fire_counts', 'context'
]

print("Indlæser LIAR test.tsv...")  
liar_test = pd.read_csv('test.tsv', sep='\t', names=liar_columns) # Indlæser LIAR testdata

fake_labels_liar = ['pants-fire', 'false', 'barely-true']   # Definerer de labels i LIAR datasættet der skal betragtes som fake news
liar_test['binary_label'] = liar_test['label'].apply(lambda x: 1 if x in fake_labels_liar else 0) # Omdanner 'label' kolonnen til en binær 'binary_label' kolonne: 1 for fake news, 0 for reliable news

print("Renser LIAR citater...")
def clean_text(text):               # Hjælperfunktion til at rense tekst: fjerner URLs, emails, tal, og specialtegn, og konverterer til lowercase
    if pd.isna(text): return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '<URL>', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '<EMAIL>', text)
    text = re.sub(r'[0-9]+[a-zA-Z]+', '<DATE>', text)
    text = re.sub(r'[0-9]+', '<NUM>', text)
    text = re.sub(r'[^a-z\s<>]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

liar_test['cleaned'] = liar_test['statement'].apply(clean_text) # Anvender rensningen på 'statement' kolonnen og gemmer det i en ny 'cleaned' kolonne

print("Stemmer LIAR citater...")
ps = PorterStemmer()
liar_test['stemmed'] = liar_test['cleaned'].apply( 
    lambda x: " ".join([ps.stem(word) for word in x.split()])   # Stammer ordene i 'cleaned' kolonnen og gemmer det i en ny 'stemmed' kolonne
)


print("\n Test på Liar datasættet med baseline model (Logistic Regression) ")
# Vektoriser LIAR med Simple Models vectorizer
X_liar_count = vectorizer.transform(liar_test['stemmed'])
y_pred_liar_simple = model.predict(X_liar_count)

print(classification_report(liar_test['binary_label'], y_pred_liar_simple, target_names=['Reliable (0)', 'Fake (1)']))


print("\n Test på Liar datasættet med avanceret model (Neuralt Netværk) ")
# Vektoriser LIAR med Advanced Models TF-IDF vectorizer
X_liar_tfidf = tfidf_vectorizer.transform(liar_test['stemmed'])
y_pred_liar_adv = nn_model.predict(X_liar_tfidf)

print(classification_report(liar_test['binary_label'], y_pred_liar_adv, target_names=['Reliable (0)', 'Fake (1)']))

print("\n Confusion Matrix for Neuralt Netværk på LIAR datasættet ")
cm = confusion_matrix(liar_test['binary_label'], y_pred_liar_adv)
print(cm)
print(f"Sande Reliable (TN): {cm[0][0]} | Falske Fake (FP): {cm[0][1]}") # TN = True Negatives, FP = False Positives
print(f"Falske Reliable (FN): {cm[1][0]} | Sande Fake (TP): {cm[1][1]}") # FN = False Negatives, TP = True Positives