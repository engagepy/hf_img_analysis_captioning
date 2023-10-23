import os
from dotenv import load_dotenv, find_dotenv
from transformers import pipeline
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
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
def generate_story(scenario):
    template = """
    You are a instagram image captioning expert.
    You can generate smart, witty captions for images on instagram.
    Never be offensive or excesively sarcastic or arrogant.
    Always generate hastags perfect for SEO and engagement.

    CONTEXT = {scenario}
    STORY:
"""
    prompt = PromptTemplate.from_template(template)
    
    llm = LLMChain (llm=ChatOpenAI(model_name="gpt-4", temperature=0.7), prompt=prompt, verbose=True)
    story = llm.predict(scenario=scenario)

    print(story)
    return story

generate_story(img2text("3.png"))
#text-to-speech