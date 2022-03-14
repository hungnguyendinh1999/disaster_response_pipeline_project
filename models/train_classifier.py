import sys

import re
import pickle
import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk import word_tokenize

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(engine.table_names()[0], con=engine)
    df = df.drop(df[(df['related'] == 2)].index) # temp solution to 'bad data' where 'related' is not binary

    X = df.message
    y = df.iloc[:, 4:].drop(['child_alone'],axis=1)
    return X, y, y.columns.tolist()


def tokenize(text):
    # Normalize
    text = re.sub(r"[^A-z0-9]", " ", text.lower())
    words = word_tokenize(text)
    # remove stopwords
    words = [w for w in words if w not in stopwords.words("english")]
    # Lemmatize
    lemmatizer = nltk.WordNetLemmatizer()
    return [lemmatizer.lemmatize(w).strip() for w in words]


def build_model():
    pipeline = Pipeline(
    [
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("clf", MultiOutputClassifier(RandomForestClassifier(class_weight="balanced"), n_jobs=-1)),
    ])

    parameters = {'clf__estimator__max_depth': [10, 20, None],
              'clf__estimator__min_samples_leaf': [1, 2, 4],
              'clf__estimator__n_estimators': [10, 20, 40]}

    cv = GridSearchCV(pipeline, param_grid=parameters, scoring='accuracy', verbose=1, n_jobs=-1)

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    y_test_pred = model.predict(X_test)
    print(classification_report(Y_test, y_test_pred, target_names=category_names))

    for  i, category in enumerate(Y_test.columns.values):
        print("{} -- {}".format(category, accuracy_score(Y_test.values[:,i], y_test_pred[:, i])))
    print("accuracy = {}".format(accuracy_score(Y_test, y_test_pred)))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()