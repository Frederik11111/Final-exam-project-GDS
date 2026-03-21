import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import time

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')

# load csv filen 
input_file = "995K_rows.csv"

# output den processerede fil
output_file = "processed_995K.csv"

# hvor mange chunks den skal læse ad gangern
chunk_size = 15000 


english_stopwords = set(stopwords.words('english'))
ps = PorterStemmer()

# Hjælpefunktion for cleaning
def clean_text(text):
    if pd.isna(text): 
        return ""
    # Lowercase
    text = text.lower()
    # Tags for specielle mønstre
    text = re.sub(r'https?://\S+|www\.\S+', '<URL>', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '<EMAIL>', text)
    text = re.sub(r'[0-9]+[a-zA-Z]+', '<DATE>', text)
    text = re.sub(r'[0-9]+', '<NUM>', text)
    # Fjern specialtegn (behold tags og bogstaver)
    text = re.sub(r'[^a-z\s<>]', '', text)
    # Rens whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# global statistik
global_vocab_initial = set()
global_vocab_filtered = set()
global_vocab_stemmed = set()


print(f"Starter processering af {input_file}...")
print(f"Chunk size: {chunk_size}")
print("-" * 70)

start_time = time.time()
reader = pd.read_csv(input_file, chunksize=chunk_size)

for i, chunk in enumerate(reader):
    chunk_start = time.time()
    
    # Rens og Baseline Tokenization
    chunk['content_cleaned'] = chunk['content'].apply(clean_text)
    chunk['tokens'] = chunk['content_cleaned'].apply(nltk.word_tokenize)
    
    # Opdater baseline vocab set
    for t_list in chunk['tokens']:
        global_vocab_initial.update(t_list)
    
    # Stopwords
    chunk['filtered_tokens'] = chunk['tokens'].apply(
        lambda t: [w for w in t if w not in english_stopwords]
    )
    for t_list in chunk['filtered_tokens']:
        global_vocab_filtered.update(t_list)
    
    # Stemming
    chunk['stemmed'] = chunk['filtered_tokens'].apply(
        lambda x: [ps.stem(w) if not w.startswith('<') else w for w in x]
    )
    for t_list in chunk['stemmed']:
        global_vocab_stemmed.update(t_list)
    
    # gemmer dataen
    is_first = (i == 0)
    chunk.to_csv(output_file, index=False, mode='w' if is_first else 'a', header=is_first)
    
    # status print
    chunk_duration = time.time() - chunk_start
    total_elapsed = time.time() - start_time
    rows_so_far = (i + 1) * chunk_size
    
    print(f"Chunk {i+1:3} | Rækker: {rows_so_far:7,} | Tid: {chunk_duration:5.2f}s | Total: {total_elapsed/60:5.2f} min")

# print stats
print("-" * 70)
print(f"Baseline Vocab Size: {len(global_vocab_initial):,}")
print(f"After Stopwords:     {len(global_vocab_filtered):,}")
print(f"After Stemming:      {len(global_vocab_stemmed):,}")

# Beregn reduktionsrater
red_stop = (1 - len(global_vocab_filtered)/len(global_vocab_initial)) * 100
red_stem = (1 - len(global_vocab_stemmed)/len(global_vocab_filtered)) * 100

print(f"\nReduktionsrater:")
print(f"Stopwords fjernede {red_stop:.2f}% af de unikke ord.")
print(f"Stemming fjernede   {red_stem:.2f}% af de resterende ord.")
