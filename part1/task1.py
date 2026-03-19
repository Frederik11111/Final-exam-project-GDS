import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')

# Indlæs url
url = "https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv"

# Læs url
readUrl = pd.read_csv(url)

# Hjælperfunktion til at strukturere og rense teksten. 
def clean_text(text):
    # Håndter tomme værdier 
    if pd.isna(text):
        return ""
    
    # Gør alt til små bogstaver
    text = text.lower()

    # Erstat URLs med et tag 
    text = re.sub(r'https?://\S+|www\.\S+', '<URL>', text)

    # Erstat e-mails med et tag
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '<EMAIL>', text)

    # Erstat datoer med et tag
    text = re.sub(r'[0-9]+[a-zA-Z]+', '<DATE>', text)

    # Erstat resterende tal med et tag
    text = re.sub(r'[0-9]+', '<NUM>', text)
    
    # Fjern specialtegn 
    text = re.sub(r'[^a-z\s<>]', '', text)
    
    # Fjern ekstra mellemrum, tabs og linjeskift
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Anvend rensningen på 'content' kolonnen
readUrl['content_cleaned'] = readUrl['content'].apply(clean_text)

# Tokenize den rensede tekst og gem det i en ny kolonne 'tokens'
readUrl['tokens'] = readUrl['content_cleaned'].apply(nltk.word_tokenize)

# Liste over alle ord i alle artikler
all_words_list = [word for tokens_list in readUrl['tokens'] for word in tokens_list]

# Sæt af unikke ord (vocab) og beregn vocab size
vocab = set(all_words_list)

# Længden af vocab
vocab_size = len(vocab)

# Set af engelske stopwords
english_stopwords = set(stopwords.words('english'))

# Print antal stopwords og de første 15 stopwords
print(f"Number of stopwords: {len(english_stopwords)}")
print(list(english_stopwords)[:15]) # Display first 15 stopwords

# Fjern stopwords fra tokens og gem det i en ny kolonne 'filtered_tokens'
readUrl['filtered_tokens'] = readUrl['tokens'].apply(lambda tokens: [word for word in tokens if word not in english_stopwords])
# Vis de første 5 rækker af tokens og filtered_tokens for at se forskellen
readUrl[['tokens', 'filtered_tokens']].head() 

# Liste over alle filtrerede ord i alle artikler
filtered_words_list = [word for tokens_list in readUrl['filtered_tokens'] for word in tokens_list]

# Sæt af unikke ord (vocab) og beregn vocab size
vocab_filtered = set(filtered_words_list)

# Længden af vocab
filtered_vocab_size = len(vocab_filtered)

# Stemming
ps = PorterStemmer()

# Anvend stemming på de filtrerede tokens og gem det i en ny kolonne 'stemmed'
readUrl['stemmed'] = readUrl['filtered_tokens'].apply(
    lambda x: [ps.stem(w) if not w.startswith('<') else w for w in x]
)

# Liste over alle stemmede ord i alle artikler
all_stemmed_words = [word for sublist in readUrl['stemmed'] for word in sublist]

# Sæt af unikke stemmede ord (vocab) og beregn vocab size
vocab_stemmed = set(all_stemmed_words)
vocab_stemmed_size = len(vocab_stemmed)

# Reduktion efter stopwords i forhold til baseline
reduction_stop = (1 - (filtered_vocab_size / vocab_size)) * 100

# Reduktion efter stemming i forhold til efter stopwords
reduction_stem = (1 - (vocab_stemmed_size / filtered_vocab_size)) * 100

# PRINT RESULTATER 
print(f"\n--- STATISTIK FOR TASK 1 ---")
print(f"Oprindelig Vocab Size: {vocab_size}")
print(f"Vocab Size efter Stopwords: {filtered_vocab_size} (Reduktion: {reduction_stop:.2f}%)")
print(f"Vocab Size efter Stemming: {vocab_stemmed_size} (Reduktion: {reduction_stem:.2f}%)")


