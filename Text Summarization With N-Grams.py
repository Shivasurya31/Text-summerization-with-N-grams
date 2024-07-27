import nltk
from nltk.util import ngrams
from collections import Counter

# Download necessary NLTK data
nltk.download('punkt')

def summarize_text(text, n=2, num_sentences=1):
    """
    Summarize the input text using n-grams.
    
    Args:
    text (str): The text to summarize.
    n (int): The size of the n-grams. Default is 2 (bigrams).
    num_sentences (int): Number of sentences to include in the summary. Default is 1.
    
    Returns:
    str: The summary of the text.
    """
    # Tokenize text
    tokens = nltk.word_tokenize(text)

    # Generate n-grams
    n_grams = list(ngrams(tokens, n))

    # Frequency analysis
    n_gram_freq = Counter(n_grams)

    # Score sentences
    sentences = nltk.sent_tokenize(text)
    sentence_scores = {}
    for sentence in sentences:
        sentence_tokens = nltk.word_tokenize(sentence)
        sentence_n_grams = list(ngrams(sentence_tokens, n))
        score = sum(n_gram_freq[n_gram] for n_gram in sentence_n_grams)
        sentence_scores[sentence] = score

    # Generate summary
    summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    summary = ' '.join(summary_sentences[:num_sentences])

    return summary

def main():
    # Take input from the user
    user_text = input("Please enter the text you want to summarize: ")
    
    # Optional: Allow user to specify n-gram size and number of summary sentences
    n = int(input("Enter the size of n-grams (e.g., 1 for unigrams, 2 for bigrams): ") or 2)
    num_sentences = int(input("Enter the number of sentences for the summary: ") or 1)

    # Generate the summary
    summary = summarize_text(user_text, n, num_sentences)

    # Print the summary
    print("\nSummary:")
    print(summary)

if name == "main":
    main()