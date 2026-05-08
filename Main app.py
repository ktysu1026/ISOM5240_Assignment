# Import part
import streamlit as st
from PIL import Image
from transformers import pipeline
import gtts
from io import BytesIO
import base64

# Function part
# img2text
# def img2text(url):
    #image_to_text_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    #text = image_to_text_model(url)[0]["generated_text"]
    #return text
# text2story
# def text2story(text):
    # story_text = ""   
    # return story_text
# def text2audio(story_text):
    #audio_data = ""    
    #return audio_data

# Image to Text Function
def get_img2text_pipe():
    # Using 'image-to-text' which is the standard registry for this model
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

def img2text(image_obj):
    pipe = get_img2text_pipe()
    result = pipe(image_obj)
    # Returns the caption generated from the image [cite: 33]
    return result[0]["generated_text"]
    
def get_story_pipe():
    # Efficient model for Streamlit Cloud deployment [cite: 36]
    return pipeline("text-generation", model="gpt2")

def text2story(description, age_range):
    generator = get_story_pipe()
    
    # Custom prompts to satisfy the 3-10 year old age range requirement [cite: 20, 21]
    prompts = {
        "3-4": f"Write a very simple, happy story for a toddler about: {description}. Use tiny words.",
        "5-6": f"Write a fun adventure story for a young child about: {description}.",
        "7+": f"Write an exciting story for a school kid about: {description}. Make it cool!"
    }
    
    prompt = prompts.get(age_range, prompts["5-6"])
    
    # Setting constraints to meet the 50-100 word requirement [cite: 27]
    # max_new_tokens is set to handle the 100-word limit safely
    story_output = generator(prompt, max_new_tokens=100, min_new_tokens=60, temperature=0.8, do_sample=True)
    story_text = story_output[0]['generated_text'].replace(prompt, "").strip()
    return story_text

# text2audio
def text2audio(story_text):
    # Converts story to audio format for an engaging experience [cite: 29, 38]
    tts = gTTS(text=story_text, lang='en', slow=False)
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    return audio_fp

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
