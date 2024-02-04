import pandas as pd
import streamlit as st
from Pipeline.data_processing_pipeline import process_data, get_csv_download_link
from Pipeline.model_training_pipeline import main as model_training_main
from Pipeline.prediction_pipeline import main as prediction_pipeline_main

def main():
    st.title("Complete Data Processing, Model Training, and Prediction App")

    # Sidebar navigation
    page = st.sidebar.selectbox("Select a page", ["Data Preprocessing", "Model Training", "Prediction"], index= 2)

    if page == "Data Preprocessing":
        st.markdown("### Data Preprocessing")
        data_preprocessing_page()
    elif page == "Model Training":
        st.markdown("### Model Training")
        model_training_main()
    elif page == "Prediction":
        st.markdown("### Model Prediction")
        prediction_pipeline_main()

def data_preprocessing_page():
    st.write(
        "This page allows you to upload raw data, preprocess it, and download the processed data."
    )

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv", "xls", "xlsx"])

    if uploaded_file is not None:
        processing_spinner = st.empty()
        processing_spinner.text("Processing uploaded data...")

        try:
            processed_features, processed_target = process_data(uploaded_file)
            processing_spinner.empty()

            st.subheader("Processed Data:")
            st.dataframe(
                pd.concat([processed_features, processed_target], axis=1).head().style.set_table_styles(
                    [{'selector': 'thead th', 'props': [('max-width', '50px')]}]
                )
            )

            st.markdown(get_csv_download_link(processed_features, processed_target), unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")

if __name__ == "__main__":
    main()
