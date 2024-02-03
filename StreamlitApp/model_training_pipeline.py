import pickle
import pandas as pd
import streamlit as st
from utils.unpickle_file import unpickle_file

RFC = unpickle_file('artifacts\\model.pkl')

def data_split(raw_dataframe):
    selected_columns = ['epoch', 'e', 'i', 'om', 'w', 'ma', 'n', 'class', 'rms', 'neo']
    dataframe = raw_dataframe[selected_columns]
    features, target = dataframe.drop('neo', axis=1), dataframe['neo']
    return features, target

def main():
    st.title("RFC Model Training App")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            st.subheader("Preview of the Uploaded Data:")
            df = pd.read_csv(uploaded_file)
            st.write(df.head())

            st.subheader("Training the Model:")
            features, target = data_split(df)
            RFC.fit(features, target)
            st.success("Model trained successfully!")

            st.subheader("Download Trained Model:")
            model_pickle = RFC 

            st.download_button(label="Download Model Pickle",
                               data=pickle.dumps(model_pickle),
                               file_name="trained_model.pkl",
                               key="trained_model")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
