import streamlit as st
from PIL import Image

# App Title and Logo (if any)
st.set_page_config(page_title="Placement Clustering App", layout="centered")

st.title("🎓 Placement Clustering App")
st.subheader("Predict Student Placement Chances with Unsupervised Learning")

# Intro text
st.markdown("""
This app uses machine learning to group students into clusters based on their placement potential:
- 🔴 Low
- 🟡 Medium
- 🟢 High

Upload your dataset, and the app will automatically assign each student to a cluster and show insightful visualizations.
""")

# Call-to-action buttons
col1, col2, col3 = st.columns(3)
with col1:
    st.link_button("🚀 Try the App",'https://placementclustering.streamlit.app/')
with col2:
    st.page_link("https://github.com/Coder15Anish/Placement_Clustering", label="📂 View Source", icon="🔗")
with col3:
    st.page_link("https://github.com/Coder15Anish/Placement_Clustering", label="📄 View Report", icon="📎")

# Optional: show an image or demo preview
# st.image("sample_cluster_plot.jpg", caption="Sample Cluster Visualization")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Anish Roy | [GitHub](https://github.com/Coder15Anish) | [LinkedIn](https://www.linkedin.com/in/anish-roy-90a477236/)")
