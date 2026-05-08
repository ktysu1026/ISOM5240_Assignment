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
    # Convert age_range (e.g., "3-4 years") to match expected format
    # Your main passes age_range which might be "3-4 years", "5-6 years", or "7+ years"
    
    # Build prompt based on age
    if age_choice == "3-4 years":
        prompt = f"child friendly story about a happy {description}. simple words: "
    elif age_choice == "5-6 years":
        prompt = f"adventure story for kids about {description}: "
    else:
        prompt = f"hero story for children about {description}: "
    
    # Load model with specific revision for stability
    generator = pipeline(
        "text-generation", 
        model="pranavpsv/genre-story-generator-v2",
        revision="main"
    )
    
    # Generate with explicit parameters
    story_results = generator(
        prompt, 
        max_new_tokens=100, 
        min_new_tokens=50, 
        do_sample=True, 
        temperature=0.7,
        truncation=True  # Add this to handle long prompts
    )
    
    # Extract story text safely
    if isinstance(story_results, list) and len(story_results) > 0:
        if isinstance(story_results[0], dict):
            generated_text = story_results[0].get('generated_text', '')
            # Remove the prompt from the generated text if it's included
            if generated_text.startswith(prompt):
                story = generated_text[len(prompt):].strip()
            else:
                story = generated_text.strip()
        else:
            story = str(story_results[0])
    else:
        story = "Once upon a time, something magical happened!"
    
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
    age_range = st.selectbox(
        "How old are you?",
        ["3-4 years", "5-6 years", "7+ years"]
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
