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
    # 1. Håndter tomme værdier (vigtigt for det store datasæt)
    if pd.isna(text):
        return ""
    
    # 2. Gør alt til små bogstaver
    text = text.lower()

    # 3. Erstat URLs med et tag (bevarer information om at der var et link)
    text = re.sub(r'https?://\S+|www\.\S+', '<URL>', text)

    # 4. Erstat e-mails med et tag
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '<EMAIL>', text)

    # 5. Erstat datoer (f.eks. 13th) med et tag
    text = re.sub(r'[0-9]+[a-zA-Z]+', '<DATE>', text)

    # 6. Erstat resterende tal med et tag
    text = re.sub(r'[0-9]+', '<NUM>', text)
    
    # 7. Fjern specialtegn (behold kun bogstaver, tags og mellemrum)
    text = re.sub(r'[^a-z\s<>]', '', text)
    
    # 8. Fjern ekstra mellemrum, tabs og linjeskift
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







