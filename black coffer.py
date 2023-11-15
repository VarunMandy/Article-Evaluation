import pandas as pd
import requests
from bs4 import BeautifulSoup

# Load the Excel file into a pandas DataFrame
input_file = 'Input.xlsx'
df = pd.read_excel(input_file)

# Create a function to scrape text from a given URL
def scrape_text_from_url(url):
    try:
        # Send an HTTP request to the URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract text content from HTML (modify as needed)
            text = soup.get_text()

            return text
        else:
            return None  # Return None if the request was not successful
    except Exception as e:
        print(f"An error occurred while scraping {url}: {str(e)}")
        return None

# Iterate through the DataFrame and scrape text from URLs
for index, row in df.iterrows():
    url_id = row['URL_ID']
    url = row['URL']

    # Call the function to scrape text from the URL
    scraped_text = scrape_text_from_url(url)

    # Save the scraped text or process it as needed (e.g., text analysis)

    # For example, you can save the text to a file
    if scraped_text:
        with open(f'{url_id}.txt', 'w', encoding='utf-8') as file:
            file.write(scraped_text)

import pandas as pd
import os
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords data (if not already downloaded)
nltk.download('stopwords')

# Load the Excel file containing scraped data
input_file = 'Input.xlsx'
df = pd.read_excel(input_file)

# Create a function to perform word frequency analysis
def word_frequency_analysis(text):
    # Tokenize the text
    words = nltk.word_tokenize(text)

    # Remove stopwords (common words like "the," "and," etc.)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.lower() not in stop_words]

    # Calculate word frequencies
    word_freq = Counter(words)

    return word_freq

# Create a folder to save the results
output_folder = 'word_frequency_results'
os.makedirs(output_folder, exist_ok=True)

# Iterate through the DataFrame and perform word frequency analysis
for index, row in df.iterrows():
    url_id = row['URL_ID']
    url = row['URL']

    # Read the scraped text from the file
    text_file = f'{url_id}.txt'
    if os.path.isfile(text_file):
        with open(text_file, 'r', encoding='utf-8') as file:
            scraped_text = file.read()

        # Perform word frequency analysis
        word_freq = word_frequency_analysis(scraped_text)

        # Save the word frequency results to a CSV file
        output_file = os.path.join(output_folder, f'{url_id}_word_frequency.csv')
        df_word_freq = pd.DataFrame(word_freq.items(), columns=['Word', 'Frequency'])
        df_word_freq.to_csv(output_file, index=False)

        print(f'Word frequency analysis for {url_id} completed and saved to {output_file}')
    else:
        print(f'No scraped text found for {url_id}')

print('Word frequency analysis process completed.')

import pandas as pd
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download NLTK VADER lexicon (if not already downloaded)
nltk.download('vader_lexicon')

# Load the Excel file containing scraped data
input_file = 'Input.xlsx'
df = pd.read_excel(input_file)

# Create a function to perform sentiment analysis
def sentiment_analysis(text):
    # Initialize the VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()

    # Analyze sentiment
    sentiment_scores = sia.polarity_scores(text)

    # Determine sentiment based on the compound score
    sentiment = 'Neutral'
    if sentiment_scores['compound'] > 0.05:
        sentiment = 'Positive'
    elif sentiment_scores['compound'] < -0.05:
        sentiment = 'Negative'

    return sentiment, sentiment_scores

# Create a folder to save the sentiment analysis results
output_folder = 'sentiment_analysis_results'
os.makedirs(output_folder, exist_ok=True)

# Iterate through the DataFrame and perform sentiment analysis
for index, row in df.iterrows():
    url_id = row['URL_ID']
    url = row['URL']

    # Read the scraped text from the file
    text_file = f'{url_id}.txt'
    if os.path.isfile(text_file):
        with open(text_file, 'r', encoding='utf-8') as file:
            scraped_text = file.read()

        # Perform sentiment analysis
        sentiment, sentiment_scores = sentiment_analysis(scraped_text)

        # Save the sentiment analysis results to a CSV file
        output_file = os.path.join(output_folder, f'{url_id}_sentiment_analysis.csv')
        df_sentiment = pd.DataFrame({'Sentiment': [sentiment], **sentiment_scores})
        df_sentiment.to_csv(output_file, index=False)

        print(f'Sentiment analysis for {url_id} completed and saved to {output_file}')
    else:
        print(f'No scraped text found for {url_id}')

print('Sentiment analysis process completed.')

pip install nltk gensim pyLDAvis
pip install bert-extractive-summarizer

import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from summarizer import Summarizer

# Download NLTK stopwords data (if not already downloaded)
nltk.download('stopwords')
nltk.download('punkt')

# Load the Excel file containing scraped data
input_file = 'Input.xlsx'
df = pd.read_excel(input_file)

# Create a function to preprocess text and generate a summary
def generate_summary(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Join the sentences back into a single text block
    text = ' '.join(sentences)

    # Generate a summary using BERT-based summarization
    summarizer = Summarizer()
    summary = summarizer(text)

    return summary

# Create a folder to save the summary results
output_folder = 'text_summarization_results'
os.makedirs(output_folder, exist_ok=True)

# Iterate through the DataFrame and generate summaries
for index, row in df.iterrows():
    url_id = row['URL_ID']
    url = row['URL']

    # Read the scraped text from the file
    text_file = f'{url_id}.txt'
    if os.path.isfile(text_file):
        with open(text_file, 'r', encoding='utf-8') as file:
            scraped_text = file.read()

        # Generate a summary
        summary = generate_summary(scraped_text)

        # Save the summary to a text file
        output_file = os.path.join(output_folder, f'{url_id}_summary.txt')
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(summary)

        print(f'Summary for {url_id} generated and saved to {output_file}')
    else:
        print(f'No scraped text found for {url_id}')

print('Text summarization process completed.')

import pandas as pd
import os
import spacy

# Load the spaCy English language model
nlp = spacy.load("en_core_web_sm")

# Load the Excel file containing scraped data
input_file = 'Input.xlsx'
df = pd.read_excel(input_file)

# Create a function to perform named entity recognition
def perform_ner(text):
    # Process the text with spaCy
    doc = nlp(text)

    # Extract named entities (e.g., persons, organizations, locations)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    return entities

# Create a folder to save the NER results
output_folder = 'ner_results'
os.makedirs(output_folder, exist_ok=True)

# Iterate through the DataFrame and perform NER
for index, row in df.iterrows():
    url_id = row['URL_ID']
    url = row['URL']

    # Read the scraped text from the file
    text_file = f'{url_id}.txt'
    if os.path.isfile(text_file):
        with open(text_file, 'r', encoding='utf-8') as file:
            scraped_text = file.read()

        # Perform NER on the text
        entities = perform_ner(scraped_text)

        # Save the NER results to a CSV file
        output_file = os.path.join(output_folder, f'{url_id}_ner_results.csv')
        df_entities = pd.DataFrame(entities, columns=['Entity', 'Label'])
        df_entities.to_csv(output_file, index=False)

        print(f'NER for {url_id} completed and saved to {output_file}')
    else:
        print(f'No scraped text found for {url_id}')

print('Named Entity Recognition (NER) process completed.')

!pip install rake-nltk

import pandas as pd
import os
from rake_nltk import Rake
import nltk
nltk.download('punkt')

# Load the Excel file containing scraped data
input_file = 'Input.xlsx'
df = pd.read_excel(input_file)

# Create a function to extract keywords from text
def extract_keywords(text):
    # Initialize the Rake keyword extractor
    r = Rake()

    # Extract keywords from the text
    r.extract_keywords_from_text(text)

    # Get the ranked keywords
    ranked_keywords = r.get_ranked_phrases()

    return ranked_keywords

# Create a folder to save the keyword extraction results
output_folder = 'keyword_extraction_results'
os.makedirs(output_folder, exist_ok=True)

# Iterate through the DataFrame and extract keywords
for index, row in df.iterrows():
    url_id = row['URL_ID']
    url = row['URL']

    # Read the scraped text from the file
    text_file = f'{url_id}.txt'
    if os.path.isfile(text_file):
        with open(text_file, 'r', encoding='utf-8') as file:
            scraped_text = file.read()

        # Extract keywords from the text
        keywords = extract_keywords(scraped_text)

        # Save the keywords to a text file
        output_file = os.path.join(output_folder, f'{url_id}_keywords.txt')
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write('\n'.join(keywords))

        print(f'Keyword extraction for {url_id} completed and saved to {output_file}')
    else:
        print(f'No scraped text found for {url_id}')

print('Keyword extraction process completed.')

!pip install pandas openpyxl textblob nltk

import pandas as pd
import os
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Download NLTK data (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')

# Load the input data
input_file = 'Input.xlsx'
df = pd.read_excel(input_file)

# Create a function to perform text analysis and compute variables
def perform_text_analysis(text):
    # Tokenize the text into words and sentences
    words = word_tokenize(text)
    sentences = sent_tokenize(text)

    # Compute variables
    word_count = len(words)
    sentence_count = len(sentences)

    # Calculate average sentence length
    avg_sentence_length = word_count / sentence_count

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]

    # Calculate percentage of complex words
    complex_word_count = len([word for word in filtered_words if len(word) > 3])  # Define complex words as words with more than 3 characters
    percentage_complex_words = (complex_word_count / word_count) * 100

    # Calculate FOG Index
    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

    # Calculate average number of words per sentence
    avg_words_per_sentence = word_count / sentence_count

    # Calculate syllables per word (approximation)
    syllable_count = sum([len(list(filter(str.isalpha, word))) for word in words])  # Count alphabetic characters as syllables
    syllable_per_word = syllable_count / word_count

    # Calculate personal pronoun count
    personal_pronouns = ['I', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves']
    personal_pronoun_count = len([word for word in words if word.lower() in personal_pronouns])

    # Calculate average word length
    avg_word_length = sum(len(word) for word in words) / word_count

    # Perform sentiment analysis
    blob = TextBlob(text)
    polarity_score = blob.sentiment.polarity
    subjectivity_score = blob.sentiment.subjectivity
    positive_score = len([sent for sent in blob.sentences if sent.sentiment.polarity > 0])
    negative_score = len([sent for sent in blob.sentences if sent.sentiment.polarity < 0])

    return positive_score, negative_score, polarity_score, subjectivity_score, avg_sentence_length, percentage_complex_words, fog_index, avg_words_per_sentence, complex_word_count, word_count, syllable_per_word, personal_pronoun_count, avg_word_length

# Create a list to store the results
results = []

# Iterate through the DataFrame and perform text analysis
for index, row in df.iterrows():
    url_id = row['URL_ID']
    url = row['URL']

    # Read the scraped text from the file
    text_file = f'{url_id}.txt'
    if os.path.isfile(text_file):
        with open(text_file, 'r', encoding='utf-8') as file:
            scraped_text = file.read()

        # Perform text analysis on the text
        analysis_result = perform_text_analysis(scraped_text)

        # Create a dictionary to store the results
        result_dict = {
            'URL_ID': url_id,
            'URL': url,
            'POSITIVE SCORE': analysis_result[0],
            'NEGATIVE SCORE': analysis_result[1],
            'POLARITY SCORE': analysis_result[2],
            'SUBJECTIVITY SCORE': analysis_result[3],
            'AVG SENTENCE LENGTH': analysis_result[4],
            'PERCENTAGE OF COMPLEX WORDS': analysis_result[5],
            'FOG INDEX': analysis_result[6],
            'AVG NUMBER OF WORDS PER SENTENCE': analysis_result[7],
            'COMPLEX WORD COUNT': analysis_result[8],
            'WORD COUNT': analysis_result[9],
            'SYLLABLE PER WORD': analysis_result[10],
            'PERSONAL PRONOUNS': analysis_result[11],
            'AVG WORD LENGTH': analysis_result[12]
        }

        results.append(result_dict)

# Create a DataFrame from the results and save it to an Excel file
output_df = pd.DataFrame(results)
output_file = 'Output Data Structure.xlsx'
output_df.to_excel(output_file, index=False, engine='openpyxl')

print('Text analysis and variable computation process completed. Results saved to', output_file)

