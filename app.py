import os
import logging
from dotenv import load_dotenv, find_dotenv
from transformers import pipeline, VitsTokenizer, VitsModel, set_seed, logging as transformers_logging
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import torch
import scipy
import streamlit as st

logging.basicConfig(level=logging.ERROR)
transformers_logging.set_verbosity_error()

load_dotenv(find_dotenv())

key_ai = os.getenv("OPAI")

#img2text

st.set_page_config(page_title="turtl.ai", page_icon="")

@st.cache_resource(show_spinner=True, max_entries=10, ttl=3600)
def use_pipe(task, model_name):
    st.success("Created pipe")
    return pipeline(f"{task}", model=f"{model_name}")

use_pipe_cache_resource = use_pipe("image-to-text", "Salesforce/blip-image-captioning-base")
 #image-to-text, Salesforce/blip-image-captioning-base
@st.cache_data()
def img2text(url):
    image_to_text = use_pipe_cache_resource
    text = image_to_text(url, max_new_tokens=100)[0]#['generated_text']
    print(text)
    #st.success("Created text")
    return text

#llm
# Add the line commented out below to generate_story template functiion that follows for hashtag generation.
#Always generate hastags perfect for SEO and engagement.

@st.cache_resource(show_spinner=True, max_entries=10, ttl=3600)
def use_model_llm(type, model_name:str, _prompt: str):
    st.success("Created Chat Model")
    return LLMChain (llm=type(model_name=f"{model_name}", temperature=0.7), prompt=_prompt, verbose=True)

@st.cache_data()
def generate_story(scenario):
    template = """
    You are a instagram image captioning expert.
    You can generate smart, witty captions for images on instagram.
    Never be offensive or excesively sarcastic or arrogant.
    

    CONTEXT = {scenario}
    STORY:
"""
    _prompt = PromptTemplate.from_template(template)
    llm = use_model_llm(ChatOpenAI, "gpt-4", _prompt) #use_model(ChatOpenAI, "gpt-4", prompt)
    story = llm.predict(scenario=scenario)

    print(story)
    #st.success("Created Story")
    return story



#text-to-speech
# Use a pipeline as a high-level helper

@st.cache_resource(show_spinner=True, max_entries=10, ttl=3600)
def use_token(model_name):
    st.success("Created tokenizer")
    return VitsTokenizer.from_pretrained(f"{model_name}")

use_token_cache_resource = use_token("facebook/mms-tts-eng")


@st.cache_resource(show_spinner=True, max_entries=10, ttl=3600)
def use_model(model_name):
    st.success("Created model")
    return VitsModel.from_pretrained(f"{model_name}")

use_model_cache_resource = use_model("facebook/mms-tts-eng")

@st.cache_data()
def text_to_speech(text):
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

    # Write the waveform to a .wav file
    scipy.io.wavfile.write("speech.wav", rate=sampling_rate, data=waveform_np)

    st.success("Created audio") 
    #os.system('afplay speech.wav')

    return waveform


#text_to_speech(generate_story(img2text("Beaches-of-Havelock...-1024x769.jpg")))

#You can fine-tune the `waveform.numpy() * <123123>` parameter. While doing so resort to function below.

#text_to_speech("Let us test this line out loud. There is a question ? A moment, and a time to speak fast so that one can get the meaning across !")

def main():
    image_file_types = ["jpg", "jpeg", "png"]

    
    st.header("Ai Image Captioning and Text to Speech by ZP")
    
    uploaded_file = st.file_uploader("Choose an image...", type=image_file_types)
    
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
            st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        
        scenario = img2text(uploaded_file.name)
        story = generate_story(scenario)
        text_to_speech(story)

        with st.expander("scenario"):
            st.write(scenario)

        with st.expander("caption"):
            st.write(story)
    
        audio_file = 'speech.wav'

        if os.path.isfile(audio_file):
            st.audio(audio_file, format='audio/wav')

        #st.success("Run complete [✔") 
        st.cache_data.clear()

if __name__=='__main__':
    main()
