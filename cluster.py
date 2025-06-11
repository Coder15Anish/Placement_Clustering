from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_cluster(df,n,features) :

    try:

        df_encoded=df[features]
        cat_col=df_encoded.select_dtypes(include=['object']).columns
        label=LabelEncoder()
        for col in cat_col:
            df_encoded[col]=label.fit_transform(df[col])

        df_scaled=df_encoded[features]
        nu_col=df_scaled.select_dtypes(include=['float64','int64']).columns
        scaler=StandardScaler()
        df_scaled[nu_col]=scaler.fit_transform(df_encoded[nu_col])


        km=KMeans(n_clusters=3,random_state=42)
        X=df_scaled
        df['Cluster']=df_encoded['Cluster']=km.fit_predict(X)

        # Analyze cluster means (based on CGPA or composite score)
        cluster_means = df_encoded.groupby('Cluster')[features].mean().mean(axis=1)
        ranking = cluster_means.sort_values(ascending=False)

        # Assign labels
        placement_labels = {
            ranking.index[0]: 'High',
            ranking.index[1]: 'Medium',
            ranking.index[2]: 'Low'
        }
        df['PlacementChance']=df_encoded['PlacementChance'] = df_encoded['Cluster'].map(placement_labels)

    except Exception as e:
        return e
 
    return df

def calculate_iq(df):
    
    try:

        df_iq=df
        df_iq['IQ']=(df['AptitudeTestScore']/4)*df['SoftSkillsRating']


    except Exception as e:
        return e    
    return df_iq

def plot(df,features):
    
    try:

        df_encoded=df[features]
        cat_col=df_encoded.select_dtypes(include=['object']).columns
        label=LabelEncoder()
        for col in cat_col:
            df_encoded[col]=label.fit_transform(df[col])

        X=df_encoded[features]
        nu_col=X.select_dtypes(include=['float64','int64']).columns
        scaler=StandardScaler()
        X[nu_col]=scaler.fit_transform(df_encoded[nu_col])


        pca=PCA(n_components=2)
        X=pca.fit_transform(X)
        


        df['pca1']=X[:,0]
        df['pca2']=X[:,1]

        # Plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='pca1', y='pca2', hue='Cluster', palette='viridis', data=df, legend="full")
        plt.title("Cluster Visualization")
        plt.legend(title="Cluster")
        
        return plt
    
    except Exception as e:
        return e     

