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
    if pd.isna(text):
        return ""
    
    # 1. Gør alt til små bogstaver
    text = text.lower()
    
    # 2. Fjern URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 3. Fjern tal og specialtegn (behold kun bogstaver og mellemrum)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 4. Fjern ekstra mellemrum
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


print(f"Antal tokens i alt (alle ord i alle artikler): {len(all_words_list)}")
print(f"Vocabulary size (unikke ord): {vocab_size}")
print(f"10 ord i the vocab (bare lige for at se): {list(vocab)[:10]}")







