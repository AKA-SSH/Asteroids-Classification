import pandas as pd
import streamlit as st
from utils.unpickle_file import unpickle_file
import base64  # Import base64 module for encoding

# Function to load the model
RFC = unpickle_file('artifacts\\model.pkl')

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
        # Display uploaded dataframe
        st.subheader("Uploaded Dataframe:")
        try:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())

            # Make predictions
            st.subheader("Predictions:")
            predictions = model_predict(uploaded_file)
            if predictions is not None:
                df['Prediction'] = predictions
                st.dataframe(df)

                # Download predictions as CSV
                st.subheader("Download Predictions:")
                csv_download_link(df, "predictions.csv", "Download Predictions CSV")
            else:
                st.warning("The uploaded file is empty or does not contain any columns.")
        except pd.errors.EmptyDataError:
            st.warning("The uploaded file is empty or does not contain any columns.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Print debug information
    st.write(f"uploaded_file: {uploaded_file}")
    st.write(f"df: {df}")

# Function to create a download link for a dataframe
def csv_download_link(df, filename, link_text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
