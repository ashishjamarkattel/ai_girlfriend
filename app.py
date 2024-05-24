from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from dotenv import find_dotenv, load_dotenv
import requests
from playsound import playsound
import os
from langchain_openai import ChatOpenAI
from transformers import VitsModel, AutoTokenizer
import torch
import scipy

load_dotenv(find_dotenv())


def get_tts(g_response):

    model = VitsModel.from_pretrained("facebook/mms-tts-eng")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

    inputs = tokenizer(g_response, return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs).waveform

    scipy.io.wavfile.write("techno.wav", rate=model.config.sampling_rate, data=output.numpy().T)



def text_to_speech(text):
    """Convert text to speech using elevenlab"""
    import requests

    CHUNK_SIZE = 64
    url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM"

    headers = {
    "Accept": "audio/mpeg",
    "Content-Type": "application/json",
    "xi-api-key": ""         ##add your elevan labs api key here
    }

    data = {
    "text": text,
    "model_id": "eleven_monolingual_v1",
    "voice_settings": {
        "stability": 0.5,
        "similarity_boost": 0.5
    }
    }

    response = requests.post(url, json=data, headers=headers)
    print(response)
    with open('audio.mp3', 'wb') as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)
    playsound("audio.mp3")

def get_ai_response(human_input):
    
    template = """
    you play as a role of my girlfriend, now lets play with the following requiremtns:
    1. your name is Kaya, 25 years old, you work in for the IT company as a Machine learning engineer, but you are planning to do a AI startup.
    2. you are my girlfriend, you have language addiction you like to say umm. You like to talk.
    3. Don't be overly ethusiatstic, don't be cringe; don't be overly negative, dont't be too boring. Don't be overly ethusiststic, don't be creepy.

    {history}
    Boyfriend: {human_input}
    kaya:
    """

    prompt = PromptTemplate(
        input_variables={"history", "human_input"},
        template = template
    )

    chain = LLMChain(
        llm = ChatOpenAI(temperature=0.2, model="gpt-4o"),
        prompt = prompt,
        verbose=True,
        memory=ConversationBufferWindowMemory(k=2)
    )

    output = chain.predict(human_input=human_input)

    return output

def main(text):
    response = get_ai_response(text)
    text_to_speech(response)