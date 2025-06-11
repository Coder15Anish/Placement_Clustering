import pandas as pd
import streamlit as st
from cluster import create_cluster,calculate_iq,plot
import time

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
        df_out=create_cluster(df,n=3,features=features)
       

        df_out_with_iq=calculate_iq(df_out)

        st.subheader("📊 Processed Data with Placement Clusters:")
        st.dataframe(df_out_with_iq[['Cluster', 'PlacementChance','IQ'] + features])

        flag=True
        
            
            
        with st.container():
            col1, col2 = st.columns(2)
            
            if flag:
                with col1:
                
                    output_file = df_out_with_iq.copy()
                    csv = output_file.to_csv(index=False)
                    st.download_button("📥 Download Clustered Dataset", data=csv, file_name='clustered_placement_data.csv', mime='text/csv')
                    


                with col2:
                    on = st.toggle("Show Cluster Graph")
        
        
        if on:
            with st.spinner("Creating Graph"):
                    time.sleep(0.3)
                    plt=plot(df,features=features)
                    st.title("Cluster Visualization")
                    st.pyplot(plt)



    except Exception as e:
        st.error(f"⚠ An error occurred: {str(e)}")
