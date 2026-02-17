import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Download nødvendige pakker fra NLTK
nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text):
    # Make all words lowercased
    text = text.lower()

    # Remove multiple white spaces, tabs and new lines
    text = re.sub(r'\s+', ' ', text).strip()

    # Replace dates (focusing on dates like 13th, as seen in the columns above) with "<DATE>"
    text = re.sub(r'[0-9]+[a-zA-Z]+', '<DATE>', text)

    # replace numbers with "<NUM>"
    text = re.sub(r'[0-9]+', '<NUM>', text)

    # Replace email addresses with "<EMAIL>"
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '<EMAIL>', text)

    # Replace URLs with "<URL>"
    text = re.sub(r'http\S+|www\.\S+', '<URL>', text)

    return text


