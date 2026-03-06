import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from collections import Counter
import matplotlib.pyplot as plt

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


# Find and print for task 3
num_urls = readUrl['content_cleaned'].str.count('<URL>').sum()
print("Number of URLs:", num_urls)

num_dates = readUrl['content_cleaned'].str.count('<DATE>').sum()
print("Number of dates:", num_dates)

num_numbers = readUrl['content_cleaned'].str.count('<NUM>').sum()
print("Number of numeric values:", num_numbers)

word_counts = Counter(all_words_list)

top100 = word_counts.most_common(100)

print("Top 100 words before cleaning:")
for word, freq in top100:
    print(word, freq)



# Plot the frequency distribution before cleaning
top10000 = word_counts.most_common(10000)

words = [w for w, f in top10000]
freqs = [f for w, f in top10000]

plt.figure(figsize=(10,5))
plt.plot(freqs)
plt.title("Frequency distribution of the 10,000 most frequent words")
plt.xlabel("Word Rank")
plt.ylabel("Frequency")
plt.show()


# Print top 100 words after cleaning
stemmed_counts = Counter(all_stemmed_words)

top100_stemmed = stemmed_counts.most_common(100)

print("Top 100 words after stemming:")
print(top100_stemmed)



#Plot the frequency distribution after cleaning
stemmed_counts = Counter(all_stemmed_words)

top100_stemmed = stemmed_counts.most_common(100)

print("Top 100 words after stemming:")
print(top100_stemmed)

top10000_stemmed = stemmed_counts.most_common(10000)
freqs_stemmed = [f for w, f in top10000_stemmed]

plt.figure(figsize=(10,5))
plt.plot(freqs_stemmed)
plt.title("Frequency after stemming")
plt.xlabel("Word Rank")
plt.ylabel("Frequency")
plt.show()
