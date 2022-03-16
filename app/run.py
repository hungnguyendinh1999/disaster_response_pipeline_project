import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    graphs = []
    
    # GRAPH 1: Distribution of Labels
    label_counts = df.iloc[:, 4:].sum().sort_values(ascending=False)
    labels = list(label_counts.index)

    graphs.append(
        {
            'data': [
                Bar(
                    x=labels,
                    y=label_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Labels',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Labels"
                }
            }
        }
    )

    # GRAPH 2: Distribution of Labels groupby 'genre'
    labels_by_genre = df.loc[:, 'genre':].groupby('genre').sum()
    labels = list(labels_by_genre.iloc[0].index)

    graphs.append(
        {
            'data': [
                Bar(
                    x=labels,
                    y=labels_by_genre.iloc[0],
                    name=labels_by_genre.index[0]
                ),
                Bar(
                    x=labels,
                    y=labels_by_genre.iloc[1],
                    name=labels_by_genre.index[1]
                ),
                Bar(
                    x=labels,
                    y=labels_by_genre.iloc[2],
                    name=labels_by_genre.index[2]
                )
            ],

            'layout': {
                'title': 'Distribution of Labels by Genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Labels"
                },
                'barmode': 'stack'
            }
        }
    )


    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()