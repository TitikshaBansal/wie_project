from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load dataset
    data = pd.read_csv("crime.csv")

    # Preprocessing
    # Assuming preprocessing steps are done here

    # Splitting data into features and target
    X = data.drop(columns=['STATE/UT', 'DISTRICT'])
    y = data['ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY']

    # Model training
    model = DecisionTreeClassifier()
    model.fit(X, y)

    # Get state name from form submission
    state_name = request.form['state']

    # Filter data for the given state
    state_data = data[data['STATE/UT'] == state_name]

    # Predict safety ranks for districts in the given state
    predicted_ranks = model.predict(state_data.drop(columns=['STATE/UT', 'DISTRICT']))

    # Calculate accuracy
    accuracy = accuracy_score(state_data['ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY'], predicted_ranks)

    # Combine predicted ranks with districts
    ranked_districts = pd.DataFrame({'District': state_data['DISTRICT'], 'Safety Rank': predicted_ranks})

    # Sort districts by safety rank
    ranked_districts = ranked_districts.sort_values(by='Safety Rank')

    return render_template('result.html', state=state_name, districts=ranked_districts, accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)
