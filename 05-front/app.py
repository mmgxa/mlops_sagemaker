import streamlit as st
from PIL import Image
import requests
import base64
from io import BytesIO

hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """
            

# @st.cache
def predict(encoded_string):
    
    url = 'https://ypce3z6egozyx5lheevt65oztm0ufcdm.lambda-url.us-west-2.on.aws/'
    data = {'body': encoded_string}
    headers = {'Content-Type': 'application/json'}
    label = requests.post(url, json=data, headers=headers).text   
    return label

def main():
    st.set_page_config(
        page_title="EMLOv2 - Capstone - MMG",
        layout="centered",
        page_icon="🐍",
        initial_sidebar_state="expanded",
    )

    st.title("Intel Image Classifier using Lambda")
    st.subheader("Upload an image to classify it")

    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "png", "jpeg"]
    )

    if st.button("Predict"):
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            io_buffer = BytesIO()
            image.save(io_buffer, format="JPEG")
            encoded_string = base64.b64encode(io_buffer.getvalue()).decode("utf-8")
            encoded_string = "data:image/jpeg;base64," + encoded_string
            
            st.image(image, caption="Uploaded Image", use_column_width=False)
            st.write("")

                    
            try:
                with st.spinner("Predicting..."):
                    label = predict(encoded_string)
                
                    st.success(f"Prediction is \"{label}\"")
            except:
                st.error("Something went wrong. Please try again.")
        else:
            st.warning("Please upload an image.")

    

if __name__ == "__main__":
    main()
