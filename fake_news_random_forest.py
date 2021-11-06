# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 23:03:50 2021

@author: mwenr
"""

import pandas as pd
from sklearn.utils import shuffle
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns 
import scikitplot as skplt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from wordcloud import WordCloud


###############################################################################
####Preprocessing
###############################################################################

#import data from csv
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

#print(fake.shape)
#print(true.shape)

#add to track fake and real
fake['target'] = 'fake'
true['target'] = 'true'

#combine dataframes
df = pd.concat([fake, true]).reset_index(drop = True)
#shuffle data
df = shuffle(df)
df= df.reset_index(drop=True)

#remove date and head
df=df.drop(columns=['date', 'title'])


#covert to lower case
df['text'] = df['text'].apply(lambda x: x.lower())

#remove punctuation
def punctuation_removal(text):
    punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
    for ele in text: 
        if ele in punc: 
            text = text.replace(ele, " ") 
    return text

df['text'] = df['text'].apply(punctuation_removal)

#remove stop words
nltk.download('stopwords')
stop = stopwords.words('english')
df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


###############################################################################
####Data Visualization
###############################################################################

#How many articles per subject?
print(df.groupby(['subject'])['text'].count())
df.groupby(['subject'])['text'].count().plot(kind="bar",color=['lavender','lavender','purple',\
                                                               'lavender','purple','purple','purple','purple'])
plt.show()

#How many fake and real articles?
print(df.groupby(['target'])['text'].count())
df.groupby(['target'])['text'].count().plot(kind="bar",)
plt.show()


#word cloud for fake news
fake_data = df[df["target"] == "fake"]
all_words = ' '.join([text for text in fake_data.text])

wordcloud = WordCloud(width= 800, height= 500,
                          max_font_size = 110,
                          collocations = False).generate(all_words)

plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Word cloud for real news
from wordcloud import WordCloud

real_data = df[df["target"] == "true"]
all_words = ' '.join([text for text in fake_data.text])

wordcloud = WordCloud(width= 800, height= 500,
                          max_font_size = 110,
                          collocations = False).generate(all_words)

plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#most frequent words counter
from nltk import tokenize

token_space = tokenize.WhitespaceTokenizer()

def counter(text, column_text, quantity):
    all_words = ' '.join([text for text in text[column_text]])
    token_phrase = token_space.tokenize(all_words)
    frequency = nltk.FreqDist(token_phrase)
    df_frequency = pd.DataFrame({"Word": list(frequency.keys()),
                                   "Frequency": list(frequency.values())})
    df_frequency = df_frequency.nlargest(columns = "Frequency", n = quantity)
    plt.figure(figsize=(12,8))
    ax = sns.barplot(data = df_frequency, x = "Word", y = "Frequency", color = 'blue')
    ax.set(ylabel = "Count")
    plt.xticks(rotation='vertical')
    plt.show()

# Most frequent words in fake news
counter(df[df["target"] == "fake"], "text", 20)
# Most frequent words in real news
counter(df[df["target"] == "true"], "text", 20)



###############################################################################
####Modeling - Random Forest
###############################################################################

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

# Split the data
X_train,X_test,y_train,y_test = train_test_split(df['text'], df.target, test_size=0.2, random_state=42)

pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', RandomForestClassifier(n_estimators=50, criterion="entropy"))])

model = pipe.fit(X_train, y_train)
prediction = model.predict(X_test)

cm = confusion_matrix(y_test, prediction)
#plot_confusion_matrix(cm, classes=['Fake', 'Real'])
print('\n Confusion Matrix: \n', cm)
print('Accuracy:',accuracy_score(y_test, prediction))
print('********************************************************')
print('\n')
print('Random Forest Model Report\n', classification_report(y_test, prediction))



skplt.metrics.plot_confusion_matrix(y_test,prediction,normalize=True)
plt.show()
