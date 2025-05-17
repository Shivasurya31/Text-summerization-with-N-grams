import numpy as np
import pandas as pd
import warnings
import re
import nltk
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
from textblob import TextBlob
import string
from string import punctuation
from nltk.corpus import stopwords
from heapq import nlargest
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt

# Download NLTK data if not already present (uncomment if running first time)
# nltk.download('punkt')
# nltk.download('stopwords')

# Setup
stop_words = set(stopwords.words('english'))
extra_punctuation = '\n—“,”‘-’'
all_punctuation = punctuation + extra_punctuation
warnings.filterwarnings('ignore')

# Contractions dictionary
contractions_dict = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don’t": "do not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "o'clock": "of the clock",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what's": "what is",
    "when's": "when is",
    "where's": "where is",
    "who's": "who is",
    "won't": "will not",
    "would've": "would have",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
    "y'all": "you all",
    # Add more contractions as needed
}
# Compile contraction regex once
contractions_re = re.compile('(%s)' % '|'.join(map(re.escape, contractions_dict.keys())))

def expand_contractions(text):
    """
    Expand contractions in the text using contractions_dict.
    """
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, text)

def clean_html(raw_html):
    """
    Remove HTML tags from text.
    """
    cleanr = re.compile('<.*?>')
    return re.sub(cleanr, '', raw_html)

def preprocess_series(article_series):
    """
    Preprocess a pandas Series of articles:
    - Lowercasing
    - remove html, urls, email ids
    - expand contractions
    - remove possessives
    - remove punctuation and stopwords
    """
    processed = article_series.str.lower()
    processed = processed.apply(clean_html)
    processed = processed.apply(lambda x: re.sub(r'\S+@\S+', '', x))  # Remove emails
    processed = processed.apply(lambda x: re.sub(r'((http|https|ftp)://\S+|www\.\S+)', '', x))  # Remove URLs
    processed = processed.apply(lambda x: x.replace('\xa0', ' '))
    processed = processed.apply(expand_contractions)
    processed = processed.apply(lambda x: re.sub(r"('s|’s)", '', x))  # remove possessives
    processed = processed.apply(lambda x: re.sub(r'\s+', ' ', x.strip()))  # remove multiple spaces
    processed = processed.apply(lambda x: ''.join(ch for ch in x if ch not in all_punctuation))  # remove punctuation
    processed = processed.apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))  # remove stopwords
    return processed

def tokenize_sentences(article_series):
    """
    Tokenize each article string into list of sentences.
    Returns list of list of sentences.
    """
    return [sent_tokenize(article) for article in article_series]

def compute_word_frequencies(sentences_list):
    """
    Compute word frequency dictionaries for list of sentence lists.
    Returns list of frequency dicts.
    """
    freq_list = []
    for sentences in sentences_list:
        freq = {}
        for sent in sentences:
            for word in word_tokenize(sent):
                freq[word] = freq.get(word,0) + 1
        # Normalize frequencies max=1
        max_freq = max(freq.values()) if freq else 1
        for word in freq:
            freq[word] /= max_freq
        freq_list.append(freq)
    return freq_list

def score_sentences(sentences_list, freq_list):
    """
    Score sentences based on word frequencies.
    Returns list of dictionaries {sentence: score}
    """
    all_scores = []
    for sentences, freq in zip(sentences_list, freq_list):
        sent_scores = {}
        for sent in sentences:
            for word in word_tokenize(sent):
                if word in freq:
                    sent_scores[sent] = sent_scores.get(sent, 0) + freq[word]
        all_scores.append(sent_scores)
    return all_scores

def summarize_articles(sent_score_list, ratio=0.25):
    """
    Generate summaries by selecting top sentences by score.
    ratio defines fraction of sentences to select.
    Returns list of summaries as strings.
    """
    summaries = []
    for sent_score in sent_score_list:
        if not sent_score:
            summaries.append('')
            continue
        select_count = max(1, int(len(sent_score)*ratio))
        top_sents = nlargest(select_count, sent_score, key=sent_score.get)
        # Join sentences with proper spacing
        summary = '. '.join(top_sents).strip()
        if not summary.endswith('.'):
            summary += '.'
        summaries.append(summary)
    return summaries

def article_summarize(input_data):
    """
    Main function to summarize articles.
    Accepts string, list of strings or pandas Series.
    Returns list of summaries.
    """
    # Convert to pandas Series if not already
    if not isinstance(input_data, pd.Series):
        input_data = pd.Series(input_data) if isinstance(input_data, (list, np.ndarray)) else pd.Series([input_data])
    
    preprocessed = preprocess_series(input_data)
    sentences_list = tokenize_sentences(input_data)
    freq_list = compute_word_frequencies(sentences_list)
    sent_scores = score_sentences(sentences_list, freq_list)
    summaries = summarize_articles(sent_scores)
    return summaries

def generate_wordcloud(text, title='Word Cloud'):
    """
    Generate and show a Word Cloud from input text string.
    """
    wc = WordCloud(width=1000, height=500, background_color='white').generate(text)
    plt.figure(figsize=(15,8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=24)
    plt.show()

# Example usage (comment or adapt for your context):
if __name__=='__main__':
    # Load data (update paths as necessary)
    df1 = pd.read_csv("/kaggle/input/all-the-news/articles1.csv")
    df2 = pd.read_csv("/kaggle/input/all-the-news/articles2.csv")
    df3 = pd.read_csv("/kaggle/input/all-the-news/articles3.csv")
    df = pd.concat([df1, df2, df3], ignore_index=True)
    df.rename(columns={'content': 'article'}, inplace=True)
    
    # Sample summary of first 5 articles
    summaries = article_summarize(df['article'][:5])
    for i, summ in enumerate(summaries):
        print(f"Summary of article {i+1}:\n{summ}\n")
    
    # Generate word cloud for first article (uncomment to use)
    # generate_wordcloud(df['article'][0], title="Word Cloud for Article 1")
