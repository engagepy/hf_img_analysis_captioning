import os
import logging
from dotenv import load_dotenv, find_dotenv
from transformers import pipeline, VitsTokenizer, VitsModel, set_seed, logging as transformers_logging
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import torch
import streamlit as st

logging.basicConfig(level=logging.ERROR)
transformers_logging.set_verbosity_error()

load_dotenv(find_dotenv())
key_ai = os.getenv("OPAI")
st.set_page_config(page_title="turtl.ai", page_icon="")

#Removes streamlit footer and hamburger menu
HIDE_STREALIT_STYLE = """
            <style>
            [data-testid="stToolbar"] {visibility: hidden !important;}
            footer {visibility: hidden !important;}
            </style>
            """
st.markdown(HIDE_STREALIT_STYLE, unsafe_allow_html=True)

@st.cache_resource(show_spinner=True, ttl=24*3600)
def use_pipe(task, model_name):
    """#Cache pipe to use in image-to-text"""
    return pipeline(f"{task}", model=f"{model_name}")

use_pipe_cache_resource = use_pipe("image-to-text", "Salesforce/blip-image-captioning-base")


@st.cache_data()
def img2text(url):
    """#image-to-text, Salesforce/blip-image-captioning-base"""
    image_to_text = use_pipe_cache_resource
    text = image_to_text(url, max_new_tokens=100)[0]#['generated_text']
    print(text)
    os.remove(url)
    return text

#llm
@st.cache_resource(show_spinner=True,ttl=24*3600)
def use_model_llm(type, model_name:str, _prompt: str):
    """Cache tokenizer model for use in generate_caption"""
    return LLMChain (llm=type(model_name=f"{model_name}", temperature=0.7), prompt=_prompt, verbose=True)

# Add the line commented out below to generate_caption template functiion that follows for hashtag generation.
#Always generate hastags perfect for SEO and engagement.
@st.cache_data()
def generate_caption(scenario):
    """text from image to text is passed to ChatOpenAI model to generate a caption"""
    template = """
    You are a instagram image captioning expert.
    You can generate smart, witty captions for images on instagram.
    Never be offensive or excesively sarcastic or arrogant.
    CONTEXT = {scenario}
    STORY:
"""
    _prompt = PromptTemplate.from_template(template)
    llm = use_model_llm(ChatOpenAI, "gpt-4", _prompt) #use_model(ChatOpenAI, "gpt-3.5-turbo", prompt)
    story = llm.predict(scenario=scenario)
    print(story)
    return story

#text-to-speech

@st.cache_resource(show_spinner=True, ttl=24*3600)
def use_token(model_name):
    """Cache tokenizer for use in text-to-speech"""
    return VitsTokenizer.from_pretrained(f"{model_name}")

use_token_cache_resource = use_token("facebook/mms-tts-eng")

@st.cache_resource(show_spinner=True, ttl=24*3600)
def use_model(model_name):
    """Cache tokenizer model for use in text-to-speech"""
    st.success("Ai Pipeline, Tokenizer and Model Loaded")
    return VitsModel.from_pretrained(f"{model_name}")

use_model_cache_resource = use_model("facebook/mms-tts-eng")


@st.cache_data()
def text_to_speech(text):
    """#Audio generation happens with cache_data() observe not cache_resource()"""
    tokenizer = use_token_cache_resource
    model = use_model_cache_resource
    inputs = tokenizer(text=text, return_tensors="pt")
    set_seed(555)  # make deterministic
    with torch.no_grad():
        outputs = model(**inputs)
    waveform = outputs.waveform[0]
    sampling_rate = model.config.sampling_rate  # Get the sampling rate from model's configuration
    # Convert tensor to numpy array, rescale it to int16 range, and ensure it's in the right format
    waveform_np = (waveform.numpy() * 32767).astype('int16')
    st.success("Inference Complete [✔") 
    return waveform_np, sampling_rate

#text_to_speech(generate_caption(img2text("Beaches-of-Havelock...-1024x769.jpg")))
#You can fine-tune the `waveform.numpy() * <123123>` parameter. While doing so resort to function below.
#text_to_speech("Let us test this line out loud. There is a question ? A moment, and a time to speak fast so that one can get the meaning across !")

def main():
    """Main function to run the app"""
    
    image_file_types = ["jpg", "jpeg", "png"]
    st.header("Caption & Text2Speech by ZP")
    uploaded_file = st.file_uploader("Choose an image...", type=image_file_types)
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        scenario = img2text(uploaded_file.name)
        story = generate_caption(scenario)
        audio_file, x = text_to_speech(story)

        with st.expander("scenario"):
            st.write(scenario["generated_text"])

        with st.expander("caption"):
            st.write(story)
        st.audio(audio_file, sample_rate=x)
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.cache_data.clear()
if __name__=='__main__':
    main()
