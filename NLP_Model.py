''' Program to analyse Restaurant Review data'''

import pandas as pd
df = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Importing Regular Expression
import re
import nltk

''' Downloading the NLTK stopwords'''

nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

# List to store the cleaned text
corpus = []

''' Cleaning the Review data'''

def clean_text(i):
    # 1. Select only text
    review = re.sub('[^a-zA-z]', ' ', df['Review'][i])
    
    # 2. Convert all text to lower case
    review = review.lower()
    
    # 3. Split string
    review = review.split()
    
    # 4. Remove stopwords and perform stemming
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    
    # JOin to get a string again
    review = ' '.join(review)
    
    # 5. return review
    return review

''' Perform cleaning of data'''

for i in range(0,1000):
    corpus.append(clean_text(i))
    
''' Implementing Bag of words model '''

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

# Creating Sparse Matrix - real converion of text to Matrix
X = cv.fit_transform(corpus).toarray()

# Train test split
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X,df['Liked'], test_size = 0.2, random_state = 0)

# Fitting SVM model
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', gamma = 'scale')
classifier.fit(xtrain,ytrain)

# Evaluation
y_pred= classifier.predict(xtest)

from sklearn.metrics import accuracy_score
accuracy_score = accuracy_score(y_pred, ytest)
print(accuracy_score)

if __name__ == '__main__()':
    pass