import pandas as pd
import streamlit as st
from utils.unpickle_file import unpickle_file
import base64  # Import base64 module for encoding
import os

artifacts_dir = 'artifacts'

LE_path = os.path.join(artifacts_dir, 'label_encoder.pkl')
KM_path = os.path.join(artifacts_dir, 'kmeans_clustering.pkl')
RFC_path = os.path.join(artifacts_dir, 'model.pkl')

LE = unpickle_file(file_name=LE_path)
KM = unpickle_file(file_name=KM_path)
RFC = unpickle_file(file_name=RFC_path)

# Function to make predictions
def model_predict(test_features):
    try:
        if test_features is not None and test_features.size > 0:
            test_features = pd.read_csv(test_features)
            prediction = RFC.predict(test_features)
            return prediction
        else:
            return None
    except pd.errors.EmptyDataError:
        return None

# Streamlit App
def main():
    st.title("CSV Uploader and Model Prediction")

    # Upload CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        st.subheader("Uploaded Dataframe:")
        raw_dataframe = pd.read_csv(uploaded_file)

        # screening in only the selected columns
        selected_columns = ['epoch', 'e', 'i', 'om', 'w', 'ma', 'n', 'class', 'rms']
        dataframe = raw_dataframe[selected_columns]

        # applying label encoding
        dataframe['class'] = LE.fit_transform(dataframe['class'])

        # applying clustering
        dataframe['cluster'] = KM.fit_predict(dataframe)

        # making predictions
        predictions = RFC.predict(dataframe)
        dataframe['predicted_neo'] = predictions

        # Display the modified DataFrame
        st.dataframe(dataframe.head())

        # Download Button
        download_button = st.button("Download Predictions")

        if download_button:
            csv_data = dataframe.to_csv(index=False).encode()
            b64_csv = base64.b64encode(csv_data).decode()
            href = f'<a href="data:file/csv;base64,{b64_csv}" download="modified_dataframe.csv">Download Modified DataFrame</a>'
            st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
