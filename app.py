import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

flag=False
features = ['CGPA', 'Internships', 'Projects', 'Workshops/Certifications',
        'AptitudeTestScore', 'SoftSkillsRating', 'ExtracurricularActivities',
        'PlacementTraining', 'SSC_Marks', 'HSC_Marks']

st.title("🎓 Student Placement Clustering App (K-Means)")
st.write("Upload a dataset to cluster students into High, Medium, or Low placement chance groups.")

# File uploader
df = st.file_uploader("Upload your CSV file", type=["csv"])

if df is not None:
    df = pd.read_csv(df)

if df is not None:
    

    try:
            df_encoded=df[features]
            cat_col=df_encoded.select_dtypes(include=['object']).columns
            label=LabelEncoder()
            for col in cat_col:
                df_encoded[col]=label.fit_transform(df[col])

    except:
        st.error("Encoding Error")

            
    try:        
            df_scaled=df_encoded[features]
            nu_col=df_scaled.select_dtypes(include=['float64','int64']).columns
            scaler=StandardScaler()
            df_scaled[nu_col]=scaler.fit_transform(df_encoded[nu_col])    
    except:
        st.error("Scaling Error")

    try:
            km=KMeans(n_clusters=3,random_state=42)
            X=df_scaled
            df_encoded['cluster']=km.fit_predict(X)

            map = { 0:'Medium',
            1:'Low',
            2:'High'}
            
            df_encoded['PlacementChance']=df['PlacementChance']=df_encoded['cluster'].apply(lambda x:map.get(x,'unknown'))

            st.subheader("📊 Processed Data with Placement Clusters:")
            st.dataframe(df_encoded[['cluster', 'PlacementChance'] + features].head())
            flag=True

            if flag:
                output_file = df.copy()
                csv = output_file.to_csv(index=False)
                st.download_button("📥 Download Clustered Dataset", data=csv, file_name='clustered_placement_data.csv', mime='text/csv')




    except Exception as e:
            st.error(f"⚠ An error occurred: {str(e)}")