#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


# In[2]:


# Load dataset 
df = pd.read_csv('spam_dataset.csv')


# In[3]:


df


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.columns


# In[7]:


# to show the number of missing data
print(df.isnull().sum())


# In[8]:


df.sample(5)


# # 1. Data Cleaning
# 

# In[9]:


df.info()


# In[10]:


df.drop(columns=['Unnamed: 0','label_num'],inplace=True)


# In[11]:


df


# In[12]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[13]:


df['label'] = encoder.fit_transform(df['label'])


# In[14]:


df.head()


# In[15]:


# missing values
df.isnull().sum()


# In[16]:


df['label'].value_counts()


# In[17]:


import matplotlib.pyplot as plt
plt.pie(df['label'].value_counts(), labels=['ham','spam'],autopct="%0.2f")
plt.show()


# # Apply text preprocessing techniques 
# like tokenization, stop word removal, stemming/lemmatization, and text normalization.
# 

# In[18]:


emails = df['text']


# In[19]:


print(emails.head())


# In[20]:


from nltk.tokenize import word_tokenize

email_text = "neon retreat ho ho ho , we ' re around to that most wonderful time of the year - - - neon leaders retreat time !"
tokens = word_tokenize(email_text)
print(tokens)


# In[21]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources 
nltk.download('punkt')
nltk.download('stopwords')

email_text = "this change is needed asap for economics purposes ."

# Tokenize the email text
tokens = word_tokenize(email_text)

# Get English stopwords
stop_words = set(stopwords.words('english'))

# Remove stopwords
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

# Print the filtered tokens
print(filtered_tokens)


# In[22]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
ps.stem('needed')


# In[23]:


# Filter out stopwords
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

# Stemming using NLTK
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

print("Stemmed Tokens:", stemmed_tokens)


# In[24]:


# Convert text to lowercase
email_text = email_text.lower()

# Remove punctuation using regular expressions
email_text = re.sub(r'[^\w\s]', '', email_text)

# Print normalized text
print(email_text)


# In[25]:


# Convert text to uppercase
email_text = email_text.upper()

# Remove punctuation using regular expressions
email_text = re.sub(r'[^\w\s]', '', email_text)

# Print normalized text
print(email_text)


# In[26]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('spam_dataset.csv')

emails = df['text']
labels = df['label']

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the emails to TF-IDF feature vectors
X = tfidf_vectorizer.fit_transform(emails)

# Print the shape of the TF-IDF matrix (number of emails, number of unique words)
print("Shape of TF-IDF matrix:", X.shape)

# Print feature names (words)
print("Feature names (words):", tfidf_vectorizer.get_feature_names_out())


# In[27]:


from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer(binary=True)

# Fit and transform the emails to word presence/absence indicator feature vectors
X_word_indicator = count_vectorizer.fit_transform(emails)

print("Shape of word presence/absence indicator matrix:", X_word_indicator.shape)

print("Feature names (words):", count_vectorizer.get_feature_names_out())


# In[28]:


count_vectorizer = CountVectorizer(ngram_range=(1, 2))

# Fit and transform the emails to N-gram feature vectors
X_ngrams = count_vectorizer.fit_transform(emails)

# Print the shape of the N-gram matrix (number of emails, number of unique N-grams)
print("Shape of N-gram matrix:", X_ngrams.shape)
# Print feature names (N-grams)
print("Feature names (N-grams):", count_vectorizer.get_feature_names_out())


# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import re

df = pd.read_csv('spam_dataset.csv')

# Initialize TF-IDF vectorizer with the same parameters you used during training
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))

# Fit TF-IDF vectorizer with the dataset to learn the vocabulary
tfidf_vectorizer.fit(df['text'])

# Preprocess the email text
def preprocess_text(text):
    # Remove non-alphanumeric characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    return text

# Function to predict whether an email is spam or not spam
def predict_spam_or_not_spam(text):
    # Extract features from the preprocessed email text using TF-IDF
    X_email = tfidf_vectorizer.transform([preprocess_text(text)])
    prediction = model.predict(X_email)
    return prediction[0]

# Trained model
model = MultinomialNB()

# Fit the model with the dataset
X_train = tfidf_vectorizer.transform(df['text'])
y_train = df['label']
model.fit(X_train, y_train)

# Get email input from the user
email_index = int(input("Enter the index of the email: "))
email_text = df.loc[email_index, 'text']

# Predict whether the email is spam or not spam
prediction = predict_spam_or_not_spam(email_text)

# Display the prediction
if prediction == 'spam':
    print("The email is predicted to be spam.")
else:
    print("The email is predicted to be not spam.")


# In[ ]:





# In[ ]:





# In[ ]:




