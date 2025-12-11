import streamlit as st

def setup_page():

    st.title("Floor Layout Detection")
    st.caption("Detect floor layout from images")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        if uploaded_image is not None:
            st.image(uploaded_image)

    with col2:
        st.subheader("Detected Layout")
        if uploaded_image is not None:
            st.image(uploaded_image)
