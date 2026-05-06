# Import part
import streamlit as st
from PIL import Image
from transformers import pipeline
import gtts
from io import BytesIO
import base64

# Function part
# img2text
def img2text(url):
    image_to_text_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text_model(url)[0]["generated_text"]
    return text

# text2story
def text2story(text):
    story_text = ""   # to be completed
    return story_text

# text2audio
def text2audio(story_text):
    audio_data = ""     # to be completed
    return audio_data

# def main
def main(): 
  
    # App title
    st.title("Streamlit Demo on Hugging Face")
    
    # Write some text
    st.write("Welcome to a demo app showcasing basic Streamlit components!")
    
    # File uploader for image and audio
    uploaded_image = st.file_uploader("Upload an image",
                                      type=["jpg", "jpeg", "png"])
    
    # Display image with spinner
    if uploaded_image is not None:
        with st.spinner("Loading image..."):
            time.sleep(1)  # Simulate a delay
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Button interaction
    if st.button("Click Me"):
        st.write("🎉 You clicked the button!")

# Main part
main()
