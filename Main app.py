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

def get_story_pipe():
    # Small, efficient model for Streamlit Cloud deployment
    return pipeline("text-generation", model="gpt2")

def text2story(description, age_range):
    generator = get_story_pipe()
    
    # Prompt engineering based on age [cite: 20, 21]
    prompts = {
        "3-4": f"Write a very simple, happy 50-word story for a toddler about: {description}. Use easy words.",
        "5-6": f"Write a fun 70-word adventure story for a young child about: {description}.",
        "7+": f"Write an interesting 90-word story for a kid about: {description}. Include a small lesson."
    }
    
    prompt = prompts.get(age_range, prompts["5-6"])
    
    # Setting constraints to meet the 50-100 word requirement 
    story_output = generator(prompt, max_new_tokens=100, min_new_tokens=50, temperature=0.7, do_sample=True)
    return story_output[0]['generated_text'].replace(prompt, "").strip()

# text2story
# def text2story(text):
    # story_text = ""   
    # return story_text

# text2audio
def text2audio(story_text):
    # Using gTTS for stability on Streamlit Cloud 
    tts = gTTS(text=story_text, lang='en', slow=False)
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    return audio_fp

# def text2audio(story_text):
    #audio_data = ""    
    #return audio_data

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
