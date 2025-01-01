from flask import Flask, request, render_template
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score

# from sqlalchemy import create_engine, text
import preprocessing as pre

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

app = Flask(__name__)

# # Database connection details
# server = 'DESKTOP-PO03FEH\SQLEXPRESS'
# database = 'bca_projects'
# driver = 'ODBC Driver 17 for SQL Server'

# # Create database engine with Windows Authentication
# connection_string = f'mssql+pyodbc://{server}/{database}?driver={driver}&trusted_connection=yes'
# engine = create_engine(connection_string)

# # Corrected query to fetch data
# query = text('SELECT * FROM dbo.review_dataset')

# # Fetch data into a DataFrame using SQLAlchemy execute method
# with engine.connect() as conn:
#     result = conn.execute(query)

data = pd.read_csv('IMDB-Dataset.csv')

# Preprocessing the Data
data['sentiment'] = data['sentiment'].replace({'positive': 1, 'negative': 0})

data['review'] = data.review.apply(pre.cleanHTML)
data['review'] = data.review.apply(pre.cleanSpecialCh)
data['review'] = data.review.apply(pre.to_lower)
data['review'] = data['review'].apply(pre.cleanStopwords)
data['review'] = data['review'].apply(pre.stem_words)

# Vectorization
cv = CountVectorizer(max_features=1000)
X = cv.fit_transform(data.review)
X = X.toarray()
y = data.sentiment

# Split the data set into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

# Create and train the model
model = GaussianNB()
model.fit(X_train, y_train)
model_score = model.accuracy_score(y_test, model.predict(y_train))

@app.route('/')
def index():
    return render_template('index.html', model_score=model_score)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['review']
        
        # Preprocess the input
        clean_input = pre.cleanHTML(user_input)
        clean_input = pre.cleanSpecialCh(clean_input)
        clean_input = pre.to_lower(clean_input)
        clean_input = pre.cleanStopwords(clean_input)
        clean_input = pre.stem_words(clean_input)
        
        # Vectorize the input
        user_input_vect = cv.transform([clean_input]).toarray()
        
        # Predict the sentiment
        prediction = model.predict(user_input_vect)
        sentiment = 'positive' if prediction[0] == 1 else 'negative'
        
        return render_template('index.html', model_score=model_score, sentiment=sentiment, review=user_input)

if __name__ == '__main__':
    app.run(debug=True)
