import os
import base64
import pandas as pd
import streamlit as st
from utils.unpickle_file import unpickle_file

label_encoder_path = os.path.join('artifacts', 'label_encoder.pkl')
kmeans_clustering_path = os.path.join('artifacts', 'kmeans_clustering.pkl')
random_over_sampler_path = os.path.join('artifacts', 'random_over_sampler.pkl')

LE = unpickle_file(file_name=label_encoder_path)
KM = unpickle_file(file_name=kmeans_clustering_path)
OS = unpickle_file(file_name=random_over_sampler_path)

# Set option to display all columns
pd.set_option('display.max_columns', None)

def encode_categorical_data(features, target):
    target = target.map({'N': 0, 'Y': 1})
    categorical_columns = features.select_dtypes(include='O').columns
    for column in categorical_columns:
        features[column] = LE.fit_transform(features[column])
    return features, target

def clustering_data(features):
    features['cluster'] = KM.predict(features)
    return features

def process_data(raw_data_csv):
    raw_dataframe = pd.read_csv(raw_data_csv, low_memory=False)

    selected_columns = ['epoch', 'e', 'i', 'om', 'w', 'ma', 'n', 'class', 'rms', 'neo']
    dataframe = raw_dataframe[selected_columns]
    dataframe= dataframe.dropna()

    features, target = dataframe.drop('neo', axis=1).copy(), dataframe['neo'].copy()
    encoded_features, encoded_target = encode_categorical_data(features, target)
    engineered_features = clustering_data(encoded_features)
    resampled_features, resampled_target = OS.fit_resample(engineered_features, encoded_target)

    return resampled_features, resampled_target

def get_csv_download_link(df_features, df_target):
    # Exclude 'neo' column from df_features if it's present
    if 'neo' in df_features.columns:
        df_features = df_features.drop('neo', axis=1)
    
    # Concatenate features and target
    concatenated_df = pd.concat([df_features, df_target], axis=1)
    
    # Generate CSV file
    csv_file_str = concatenated_df.to_csv(index=False, encoding='utf-8')
    
    # Encode CSV file to base64 for download link
    b64 = base64.b64encode(csv_file_str.encode()).decode()
    
    # Create download link
    href = f'<a href="data:file/csv;base64,{b64}" download="processed_data.csv">Download processed data</a>'
    
    return href

def access_processing_pipeline():
    st.title("Data Processing App")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv", "xls", "xlsx"])

    if uploaded_file is not None:
        processing_spinner = st.empty()
        processing_spinner.text("Processing uploaded data...")
        
        try:
            processed_features, processed_target = process_data(uploaded_file)
            processing_spinner.empty()

            st.subheader("Processed Data:")
            
            # Reduce the width of the index column
            st.dataframe(pd.concat([processed_features, processed_target],
                                   axis=1).head().style.set_table_styles([{'selector': 'thead th',
                                                                           'props': [('max-width', '50px')]}]))

            st.markdown(get_csv_download_link(processed_features, processed_target), unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")

if __name__ == "__main__":
    access_processing_pipeline()
