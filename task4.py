import pandas as pd
from sklearn.model_selection import train_test_split

print("Indlæser datasættet til ML-split.")

df = pd.read_csv("processed_995K.csv", usecols=['type', 'stemmed', 'title', 'domain'])

df = df.dropna(subset=['stemmed', 'type'])

print("Omdanner tekst-labels til 0 (Reliable) og 1 (Fake)...")

fake_labels = ['fake', 'political', 'bias', 'conspiracy', 'satire', 'rumor', 'unknown', 'clickbait', 'unreliable', 'junksci', 'hate']


df['label'] = df['type'].apply(lambda x: 1 if x in fake_labels else 0)

df = df.drop(columns=['type'])

print("Udfører split (80% / 10% / 10%)...")

train_df, temp_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df['label'])

val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['label'])

print("Gemmer data til train.csv, val.csv og test.csv")
train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("\n" + "="*50)
print("Statistik:")
print("="*50)
print(f"Træningssæt (80%):    {len(train_df):,} rækker")
print(f"Valideringssæt (10%): {len(val_df):,} rækker")
print(f"Testsæt (10%):        {len(test_df):,} rækker")
print("-" * 50)
print("Tjekker splits (bør være ens for alle tre):")
print(f" - Fake News i Træning:    {(train_df['label'].mean() * 100):.1f}%")
print(f" - Fake News i Validering: {(val_df['label'].mean() * 100):.1f}%")
print(f" - Fake News i Test:       {(test_df['label'].mean() * 100):.1f}%")
print("="*50)
