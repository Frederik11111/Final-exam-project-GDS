import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import spacy

nltk.download('punkt')
nltk.download('stopwords')

filepath = 'Final-exam-project-GDS/fakenews_sample.csv'
data = pd.read_csv(filepath)

def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[0-9]+[a-zA-Z]+', '<DATE>', text)
    text = re.sub(r'[0-9]+', '<NUM>', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '<EMAIL>', text)
    text = re.sub(r'http\S+|www\.\S+', '<URL>', text)

    return text


# Rens tekstkolonne
data["cleaned_content"] = data["content"].apply(clean_text)


# Tokenize første tekst
text_example = data["cleaned_content"].iloc[0]
tokens = nltk.word_tokenize(text_example)
print("30 tokens:", tokens[:30], len(tokens))   # viser kun de første 30 tokens


# Fjern stopwords
stop_words = set(stopwords.words("english"))
filtered_tokens = [t for t in tokens if t.lower() not in stop_words]
print("30 filteren tokens:",filtered_tokens[:30], len(filtered_tokens))


# Stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(t) for t in filtered_tokens]
print("30 stemmed tokens:",stemmed_tokens[:30], len(stemmed_tokens))

#