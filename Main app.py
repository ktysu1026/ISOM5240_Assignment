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
def text2story(description, age_choice):
    generator = pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")
    
    # Matching age choice to story styles for kids
    if age_choice == "3-4 years":
        prompt = f"child friendly story about a happy {description}. simple words: "
    elif age_choice == "5-6 years":
        prompt = f"adventure story for kids about {description}: "
    else:
        prompt = f"hero story for children about {description}: "
    
    # Requirement: 50-100 words
    story_results = generator(prompt, max_new_tokens=100, min_new_tokens=50, do_sample=True, temperature=0.7)
    story = story_results[0]['generated_text'].strip()
    return story

def text2audio(story_text):
    # Using the specific model you requested
    pipe = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
    audio_output = pipe(story_text)
    return audio_output
    
# def main
def main(): 
    # Web page name
    st.set_page_config(page_title="Magic Storybook", page_icon="🎨")
    
    # App title
    # Clean UI for children
    st.title("🎨 My Magic Storybook")
    
    # Write some text
    st.write("### Upload a picture to hear a story just for you!")

    # Age selection to alter story logic
    age_range = st.select_slider(
        "How old are you?",
        options=["3-4", "5-6", "7+"]
    )
    
     # File uploader for image and audio
    uploaded_image = st.file_uploader("Choose a picture", 
                                      type=["jpg", "jpeg", "png"])
     # Display image with spinner
    if uploaded_image is not None:
        with st.spinner("Loading image..."):
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Combined processing to keep UI clean 
        if st.button("✨ Generate My Story"):
            with st.status("Magical things are happening...", expanded=False):
                # Step 1: Caption
                caption = img2text(image)
                # Step 2: Story
                story = text2story(caption, age_range)
                # Step 3: Audio
                audio_data = text2audio(story)
            
            # Display Results
            st.subheader("📖 Your Story")
            st.write(story)
            
            st.subheader("🎧 Listen")
            st.audio(audio_data, format="audio/mp3")

# Main part
if __name__ == "__main__":
    main()
