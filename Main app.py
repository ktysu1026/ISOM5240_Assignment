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

        target_words = 50

        prompt = f"""
        Write a very short and happy children's story about {description}.

Requirements:
- Use easy words for toddlers
- Exactly 5 short sentences
- Clear beginning, middle, and ending
- Everything must be safe, cheerful, and gentle
- End with everyone happy
- No scary, sad, or dangerous events

Example:
A little girl found a red ball.
The girl bounced the ball with friends.
They laughed under the sunshine.
Then they shared yummy snacks together.
Everyone smiled happily at the end.

Story:
"""

    elif age_choice == "5-6 years":

        target_words = 75

        prompt = f"""
Write a fun and magical children's story about {description}.

Requirements:
- Use simple cheerful language
- Include a beginning, middle, and ending
- The characters should face a small problem
- The problem must be solved safely
- End with everyone happy
- Avoid scary or sad events
- About {target_words} words

Example:
Tom lost his kite in the park.
He asked his friends for help.
Together they found it in a tree.
They laughed and played all afternoon.
Tom felt happy again.

Story:
"""

    else:  # 7+ years

        target_words = 120

        prompt = f"""
Write an exciting adventure story about {description}.

Requirements:
- Include friendship, bravery, and teamwork
- Have a clear beginning, middle, and ending
- Include a mystery or challenge
- The characters must solve the problem together
- Keep the story positive and child-friendly
- End with a happy and complete ending
- About {target_words} words

Example:
Mia discovered strange footprints near the old library.
She asked her friends to help solve the mystery.
Together they followed clues through the town.
At the end, they discovered a lost puppy hiding safely.
Everyone celebrated their adventure together.

Story:
"""

    # Load model with specific revision for stability
    generator = pipeline(
        "text-generation", 
        model="pranavpsv/genre-story-generator-v2",
        revision="main"
    )
    
   # Adjust max_new_tokens based on word target (roughly 1.3 tokens per word)
    max_tokens = int(target_words * 1.5)
    min_tokens = int(target_words * 0.7)
    
    # Generate with appropriate token limits for target word count
    story_results = generator(
        prompt, 
        max_new_tokens=max_tokens,
        min_new_tokens=min_tokens,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.3,
         early_stopping=True
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
    # -----------------------------
    # FALLBACK IF EMPTY
    # -----------------------------
    if not story:

        if age_choice == "3-4 years":

            story = (
            "A happy little bunny played with a red ball in the sunshine. "
            "Two friendly birds came to play too. "
            "They laughed and hopped around together. "
            "Then they shared sweet berries under a tree. "
            "Everyone smiled happily at the end."
            )

        elif age_choice == "5-6 years":

            story = (
            "One sunny morning, Mia brought her favorite teddy bear to the playground. "
            "She worried the teddy might get dirty while everyone played. "
            "Her friends helped build a tiny blanket house to keep it safe. "
            "Soon everyone played together carefully and kindly. "
            "Mia learned that sharing with good friends makes every day more fun."
            )

        else:

            story = (
            "Deep in the magical forest, four brave friends discovered glowing footprints beside an ancient tree. "
            "They followed the mysterious trail while solving fun riddles together. "
            "At the end of the path, they found a lost baby owl hiding safely inside a hollow log. "
            "Working together, they returned the owl to its family high above the forest. "
            "The friends celebrated their adventure proudly as the stars sparkled brightly overhead."
            )

    # -----------------------------
    # CLEAN FINAL OUTPUT
    # -----------------------------
    story = story.replace("\n", " ").strip()

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
            color: black !important;
            font-family: 'Fredoka', sans-serif !important;
            font-size: 1.2rem !important;
        }

        .stButton>button {
            border-radius: 20px;
            background-color: #FF4B4B;
            color: white;
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
