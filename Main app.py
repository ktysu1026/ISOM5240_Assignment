# Import part
import streamlit as st
from PIL import Image
from transformers import pipeline
from io import BytesIO
import numpy as np
from scipy.io import wavfile as wav

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
        # Focus: shorrt,
        prompt = (f"Instruction: Write a very simple toddler story about {description}.\n"
                  f"Style: Cozy, fun sounds like 'Zoom!' and 'Yay!'.\n"
                  f"Length: 5 sentences.\n\n"
                  f"Story: Once upon a time, {description}")
        target_words = 50

    elif age_choice == "5-6 years":
        # Focus: Action and simple humor
        prompt = (f"Instruction: Write a happy, safe, and magical story for a child about {description}.\n"
                  f"Constraint: No accidents, no sadness, and no scary things. Everything is bright and fun.\n"
                  f"Style: Adventurous and cheerful. Use words like 'shimmering' and 'brave'.\n"
                  f"Story Starter: It was a beautiful day for {description}. The sun was warm and everyone was happy. Then,")
        target_words = 75

    else: # 7+ years
        # Focus: Plot, problem-solving, and vivid settings
        prompt = (f"Instruction: Write an exciting adventure story about {description}.\n"
                  f"Themes: Friendship, bravery, and solving a secret mystery.\n"
                  f"Story Starter: Deep inside the heart of the {description}, a secret was waiting to be found.")
        target_words = 100
     
    # Load model with specific revision for stability
    generator = pipeline(
        "text-generation", 
        model="pranavpsv/genre-story-generator-v2",
        revision="main"
    )
    
   # Adjust max_new_tokens based on word target (roughly 1.3 tokens per word)
    max_tokens = int(target_words * 1.3)
    
    # Generate with appropriate token limits for target word count
    story_results = generator(
        prompt, 
        max_new_tokens=max_tokens,
        min_new_tokens=int(max_tokens * 0.8),
        do_sample=True, 
        temperature=0.9,  # Lower for more focused stories
        top_p=1.0,
        repetition_penalty=1.2
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
        # Fallback stories with approximate word counts based on the playground/park scene
        if age_choice == "3-4 years":
            story = f"Once upon a time, many happy children played together in a sunny park. They laughed on the swings and slid down the big slide. Everyone smiled and had so much fun playing together. The end."
        elif age_choice == "5-6 years":
            story = f"Once upon a time, a group of cheerful children played in a wonderful playground. They ran across the green grass, climbed on the jungle gym, and pushed each other on the swings. The children played tag and shared their toys with big smiles. Everyone felt happy and made new friends that day. What a beautiful day at the park it was!"
        else:  # 7+ years
            story = f"Once upon a time, many joyful children gathered at their favorite playground on a bright sunny day. The park was filled with laughter as they swung into the sky and raced down the twisting slides. Some climbed the rock wall while others played an exciting game of hide-and-seek among the trees. They took turns on the merry-go-round and helped each other up the climbing net. Every child's face glowed with happiness as they discovered that playing together made everything more fun. By sunset, they had learned that the best adventures are the ones shared with friends, and they couldn't wait to come back tomorrow."    
    return story

def text2audio(story_text):
    # Initialize pipeline
    pipe = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
    audio_output = pipe(story_text)
    
    audio_array = audio_output["audio"]
    sampling_rate = int(audio_output["sampling_rate"]) # Ensure this is an int
    
    # 1. Handle potential 2D output (squeeze to 1D if necessary)
    audio_array = np.squeeze(audio_array)
    
    # 2. Convert float32 to int16 correctly
    if audio_array.dtype != np.int16:
        # Clip to avoid wrap-around distortion before converting
        audio_array = np.clip(audio_array, -1.0, 1.0)
        audio_array = (audio_array * 32767).astype(np.int16)
    
    buffer = BytesIO()
    # Scipy expects (buffer, rate, data)
    wav.write(buffer, sampling_rate, audio_array)
    buffer.seek(0)
    return buffer

def add_custom_style():
    st.markdown(
        """
        <style>
        /* 1. Import must be at the very top */
        @import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;700&display=swap');

        /* 2. Background for the whole app */
        .stApp {
            background-color: #FFDEE9;
            background-image: linear-gradient(0deg, #FFDEE9 0%, #B5FFFC 100%);
        }

        /* Target the Main Title and the Subheader */
        h1, h3 {
            color: White !important;
            font-family: 'Fredoka', sans-serif !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2); /* Adds a soft shadow so white is easy to read */
        }

        /* Make the "Choose a picture" text white too */
        label {
            color: white !important;
            font-family: 'Fredoka', sans-serif !important;
            font-size: 1.2rem !important;
        }

        .stButton>button {
            border-radius: 20px;
            background-color: #FF4B4B;
            color: black;
            border: 2px solid #FF4B4B;
            font-weight: bold;
            font-family: 'Fredoka', sans-serif;
        }
        .stApp::before {
            content: "✨ 🦁 🎨 🌟";
            position: fixed;
            top: 20px;
            right: 20px;
            font-size: 2rem;
            opacity: 0.5;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
# def main
def main(): 
    
    # 1. Config and Style
    st.set_page_config(page_title="Magic Storybook", page_icon="🎨")
    add_custom_style() # This adds the colors and stars!
    
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
            with st.spinner("Magical things are happening..."):
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
