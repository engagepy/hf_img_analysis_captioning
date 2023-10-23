import os
from dotenv import load_dotenv, find_dotenv
from transformers import pipeline, VitsTokenizer, VitsModel, set_seed
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import torch
import scipy
load_dotenv(find_dotenv())

# Use a pipeline as a high-level helper
key_ai = os.getenv("OPAI")

#img2text
def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text(url)[0]#['generated_text']
    print(text)
    return text

#llm
# Add the line commented out below to generate_story template functiion that follows for hashtag generation.
#Always generate hastags perfect for SEO and engagement.
def generate_story(scenario):
    template = """
    You are a instagram image captioning expert.
    You can generate smart, witty captions for images on instagram.
    Never be offensive or excesively sarcastic or arrogant.
    

    CONTEXT = {scenario}
    STORY:
"""
    prompt = PromptTemplate.from_template(template)
    
    llm = LLMChain (llm=ChatOpenAI(model_name="gpt-4", temperature=0.7), prompt=prompt, verbose=True)
    story = llm.predict(scenario=scenario)

    print(story)
    return story




#text-to-speech
# Use a pipeline as a high-level helper
def text_to_speech(text):
    tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
    model = VitsModel.from_pretrained("facebook/mms-tts-eng")

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


    os.system('afplay speech.wav')

    return waveform, model

 
    

    
    
    
    # text_to_speech = pipeline("text-to-speech", model="facebook/mms-tts-eng")
    # speech = text_to_speech(text)['audio']
    # print(speech)
    # #write speech as .wav file
    # with open('speech.wav', 'wb') as f:
    #     f.write(speech)
    # return speech


text_to_speech(generate_story(img2text("4.png")))

#You can fine-tune the `waveform.numpy() * <123123>` parameter. While doing so resort to function below.

#text_to_speech("Let us test this line out loud. There is a question ? A moment, and a time to speak fast so that one can get the meaning across !")