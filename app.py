from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/district')
def district():
    # Add logic to handle district related requests
    return render_template('district.html')

@app.route('/state')
def state():
    # Add logic to handle state related requests
    return render_template('state.html')

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

@app.route('/crime_details', methods=['POST'])
def crime_details():
    data = pd.read_csv("crime.csv")
    district_name = request.form['district']

    # Filter data for the entered district
    district_data = data[data['DISTRICT'] == district_name]

    # Remove non-crime columns and sum the counts for each crime type
    crime_counts = district_data.drop(columns=['STATE/UT', 'DISTRICT']).sum()

    # Convert the crime counts to a dictionary
    crime_counts_dict = crime_counts.to_dict()

    return render_template('crime_details.html', district=district_name, crime_counts=crime_counts_dict)


if __name__ == '__main__':
    app.run(debug=True)