import pandas as pd
import re
import matplotlib.pyplot as plt
from collections import Counter

print("Indlæser datasæt (dette kan tage lidt tid)...")
df = pd.read_csv("processed_995K.csv", usecols=['type', 'domain', 'authors', 'content', 'content_cleaned', 'stemmed'])

df['content'] = df['content'].fillna('')
df['content_cleaned'] = df['content_cleaned'].fillna('')
df['stemmed'] = df['stemmed'].fillna('')

print("\n" + "="*60)
print("DEL 1: METADATA & ARTEFAKTER (URLs, Dates, Numbers)")
print("="*60)


url_count = df['content'].str.count(r'https?://\S+|www\.\S+').sum()
date_count = df['content'].str.count(r'[0-9]+[a-zA-Z]+').sum()
num_count = df['content'].str.count(r'[0-9]+').sum()

print(f"Totalt antal URL'er i datasættet:   {url_count:,.0f}")
print(f"Totalt antal Datoer i datasættet:   {date_count:,.0f}")
print(f"Totalt antal Tal-værdier:           {num_count:,.0f}")

# Frigør RAM ved at slette den rå content-kolonne, da vi ikke skal bruge den mere
del df['content']


print("\n" + "="*60)
print("DEL 2: SAMMENLIGNING AF FAKE VS. RELIABLE")
print("="*60)


fake_labels = ['fake', 'political', 'bias', 'conspiracy', 'satire', 'rumor', 'unknown', 'clickbait', 'unreliable', 'junksci', 'hate']
fake_news = df[df['type'].isin(fake_labels)]
real_news = df[df['type'] == 'reliable']

# Beregn ordantal og TTR (Type-Token Ratio)
def calculate_text_stats(text):
    words = str(text).split()
    word_count = len(words)
    if word_count == 0:
        return 0, 0.0
    ttr = len(set(words)) / word_count
    return word_count, ttr

print("Beregner ordantal og leksikalsk rigdom (TTR)...")
df['word_count'], df['ttr'] = zip(*df['content_cleaned'].apply(calculate_text_stats))

# Opdater vores fake/real dataframes med de nye kolonner
fake_news = df[df['type'].isin(fake_labels)]
real_news = df[df['type'] == 'reliable']

print(f"Gennemsnitlig artikellængde (ord):")
print(f" - Fake News:     {fake_news['word_count'].mean():.0f} ord")
print(f" - Reliable News: {real_news['word_count'].mean():.0f} ord")

print(f"\nGennemsnitlig TTR (Leksikalsk rigdom):")
print(f" - Fake News:     {fake_news['ttr'].mean():.4f}")
print(f" - Reliable News: {real_news['ttr'].mean():.4f}")

# Missing authors
missing_fake = fake_news['authors'].isnull().mean() * 100
missing_real = real_news['authors'].isnull().mean() * 100
print(f"\nArtikler uden forfatter (Anonymitet):")
print(f" - Fake News:     {missing_fake:.1f}%")
print(f" - Reliable News: {missing_real:.1f}%")

# Super-spreaders (Domæner)
fake_domains = fake_news['domain'].value_counts()
top_10_fake_pct = (fake_domains.head(10).sum() / fake_domains.sum()) * 100
print(f"\nDomæne-koncentration:")
print(f" - De top 10 mest aktive domæner producerer {top_10_fake_pct:.1f}% af al fake news!")


print("\n" + "="*60)
print("DEL 3: ORDFREKVENSER & ZIPF'S LOV")
print("="*60)

# Funktion til at tælle ord
def get_word_counts(series, is_list_string=False):
    counter = Counter()
    for text in series:
        if is_list_string:
            text = str(text).replace('[', '').replace(']', '').replace("'", "")
            words = [w.strip() for w in text.split(',') if w.strip()]
        else:
            words = str(text).split()
        counter.update(words)
    return counter

print("Tæller ord FØR stopwords & stemming...")
counter_before = get_word_counts(df['content_cleaned'], is_list_string=False)

print("Tæller ord EFTER stopwords & stemming...")
counter_after = get_word_counts(df['stemmed'], is_list_string=True)

print("\nDe 10 mest hyppige ord FØR rensning:")
print([word for word, count in counter_before.most_common(10)])

print("\nDe 10 mest hyppige ord EFTER rensning:")
print([word for word, count in counter_after.most_common(10)])

# --- PLOT ZIPF'S LOV ---
print("\nGenererer Zipf's Law plots (luk graf-vinduet for at afslutte scriptet)...")

freq_before = [count for word, count in counter_before.most_common(10000)]
freq_after = [count for word, count in counter_after.most_common(10000)]

plt.figure(figsize=(14, 6))

# Subplot 1: Før
plt.subplot(1, 2, 1)
plt.plot(freq_before, color='red', linewidth=2)
plt.title("FØR Stopwords & Stemming (Top 10.000)", fontsize=14)
plt.xlabel("Ordrang (Rank)", fontsize=12)
plt.ylabel("Frekvens (Frequency)", fontsize=12)
plt.yscale('log')
plt.xscale('log')
plt.grid(True, alpha=0.3)

# Subplot 2: Efter
plt.subplot(1, 2, 2)
plt.plot(freq_after, color='blue', linewidth=2)
plt.title("EFTER Stopwords & Stemming (Top 10.000)", fontsize=14)
plt.xlabel("Ordrang (Rank)", fontsize=12)
plt.ylabel("Frekvens (Frequency)", fontsize=12)
plt.yscale('log')
plt.xscale('log')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()