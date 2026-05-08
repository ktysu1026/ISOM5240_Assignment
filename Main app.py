# Import part
import streamlit as st
from PIL import Image
from transformers import pipeline
import gtts
from io import BytesIO

# Function part
# img2text
def img2text(url):
    image_to_text_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text_model(url)[0]["generated_text"]
    return text

# Difficulties faced during project:
# Story not finished: asked LLM, potential method to solve is to add tokens or add parameters to the generator of text2story function
# to encourage completion. 

# text2story
def text2story(description, age_choice):
    # Convert age_range (e.g., "3-4 years") to match expected format
    # Your main passes age_range which might be "3-4 years", "5-6 years", or "7+ years"
    
    # Set word count and prompt based on age
    if age_choice == "3-4 years":
        prompt = f"Tell a complete short story about a happy {description} in exactly 50 words: "
        target_words = 50
    elif age_choice == "5-6 years":
        prompt = f"Tell a complete fun story for kids about {description} in exactly 75 words: "
        target_words = 75
    else:  # "7+ years"
        prompt = f"Tell a complete interesting and adventurous story for children about {description} in exactly 100 words: "
        target_words = 100

     # Load model with specific revision for stability
    generator = pipeline(
        "text-generation", 
        model="pranavpsv/genre-story-generator-v2",
        revision="main"
    )
    
    # Generate with enough tokens for a complete story
    story_results = generator(
        prompt, 
        max_new_tokens=200,  # Generous limit for 100-150 word story
        min_new_tokens=80,
        do_sample=True, 
        temperature=0.8,  # Slightly higher for more creativity
        top_p=0.9
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
    try:
        pipe = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
        audio_output = pipe(story_text)
        
        audio_array = audio_output["audio"]
        sampling_rate = audio_output["sampling_rate"]
        
        # Convert to proper format
        if audio_array.dtype != np.int16:
            audio_array = (audio_array * 32767).astype(np.int16)
        
        buffer = BytesIO()
        wav.write(buffer, sampling_rate, audio_array)
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        # Fallback to gTTS if the pipeline fails
        print(f"Pipeline failed, using gTTS: {e}")
        from gtts import gTTS
        
        tts = gTTS(text=story_text, lang='en', slow=False)
        buffer = BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        return buffer
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
            st.audio(audio_data)

# Main part
if __name__ == "__main__":
    main()
