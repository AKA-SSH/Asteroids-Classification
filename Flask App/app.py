import os
import pandas as pd
from flask import Flask, render_template, request, send_file
from Pipeline.data_processing_pipeline import process_data, get_csv_download_link
from Pipeline.model_training_pipeline import data_split
from utils.unpickle_file import unpickle_file

app = Flask(__name__)

# Load the trained model
model_path = os.path.join('artifacts', 'model.pkl')
RFC = unpickle_file(file_name=model_path)

# Define a function for model training
def train_model(features, target):
    RFC.fit(features, target)

# Define a function for model prediction
def model_predict(test_features):
    try:
        if test_features is not None and test_features.size > 0:
            prediction = RFC.predict(test_features)
            return prediction
        else:
            return None
    except Exception as e:
        print(f"An error occurred during prediction: {str(e)}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/preprocess', methods=['GET', 'POST'])
def preprocess():
    if request.method == 'POST':
        uploaded_file = request.files['csv_file']
        if uploaded_file.filename != '':
            try:
                processed_features, processed_target = process_data(uploaded_file)
                processed_csv_link = get_csv_download_link(processed_features, processed_target)
                return render_template('preprocess.html', title='Preprocessed Data', csv_link=processed_csv_link)
            except Exception as e:
                error_message = f"Error processing data: {str(e)}"
                return render_template('preprocess.html', title='Error', error=error_message)
    
    return render_template('preprocess.html', title='Asteroid Classification')

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        uploaded_file = request.files['csv_file']
        if uploaded_file.filename != '':
            try:
                dataframe = pd.read_csv(uploaded_file)
                features, target = data_split(dataframe)
                train_model(features, target)
                return render_template('train.html', title='Training Successful', trained=True)
            except Exception as e:
                error_message = f"Error training model: {str(e)}"
                return render_template('train.html', title='Error', error=error_message)
    
    return render_template('train.html', title='Train Model')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        uploaded_file = request.files['csv_file']
        if uploaded_file.filename != '':
            try:
                # Process the uploaded file
                dataframe = pd.read_csv(uploaded_file)
                predictions = model_predict(dataframe)
                
                if predictions is not None:
                    # Add predictions to the DataFrame
                    dataframe['predicted_neo'] = predictions
                    
                    # Convert DataFrame to CSV
                    csv_data = dataframe.to_csv(index=False)
                    
                    # Return CSV file as attachment
                    return send_file(
                        filename_or_fp=csv_data,
                        mimetype='text/csv',
                        as_attachment=True,
                        attachment_filename='predicted_results.csv'
                    )
                else:
                    return render_template('error.html', error="No data for prediction.")
            except Exception as e:
                error_message = f"Error predicting data: {str(e)}"
                return render_template('error.html', error=error_message)
        else:
            return render_template('error.html', error="No file uploaded.")
    else:
        return render_template('error.html', error="Invalid request method.")


if __name__ == '__main__':
    app.run(debug=True)
