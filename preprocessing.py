# NLP libraries
from nltk.corpus import stopwords # For using the pre-built stopwords
from nltk.tokenize import word_tokenize # For tokennization
from nltk.stem import SnowballStemmer # For stemming 
import re

# Pattern for HTML tags using regex
# pattern = r'<.*?>'

# Define a function which takes text and reemoves HTML tags
def cleanHTML(text):
    cleaned = re.compile(r'<.*?>') # This will compile the regex, returns object pattern
    return re.sub(cleaned, ' ', text) # Return the text by substituting the HTML tags with blank space

# Remove the characters which are neither an alphabet nor a digit
# REGEX pattern for a character which is neither an alphabet nor a digit
pattern = r'[^a-zA-Z0-9]'

def cleanSpecialCh(text):
    return re.sub(pattern, ' ', text)

# Define a function for converting all the letters to lowercase
def to_lower(text):
    return text.lower()

# To remove the stopwords from the reviewa
# We make use of the pre-built stopwords library from :
import nltk
nltk.download('punkt')

def cleanStopwords(text):
    stopWords = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    return [tk for tk in tokens if tk not in stopWords]

# Define a function which stem the words
def stem_words(text):
    ss = SnowballStemmer('english')
    return ' '.join([ss.stem(tk) for tk in text])