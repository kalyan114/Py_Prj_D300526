#import all the necessary libraries
import os
from dotenv import load_dotenv

#this is older version, so using the latest version below for ollama langchain
# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


#load environment variables
load_dotenv()

os.environ['GEMINI_API_KEY']=os.getenv('GEMINI_API_KEY')
#langsmith tracking
os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2']="true"
os.environ['LANGCHAIN_PROJECT']=os.getenv('LANGCHAIN_PROJECT')

#prompt template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to the question asked"),
        ("user","Question:{question}")
    ]
)

#streamlit framework
# to run this app.py, after coding, run in cmd: streamlit run app.py
st.title("Langchain demo with Gemma model")
input_text=st.text_input("What question you have in mind?")


#ollama gemma:2b model
llm=OllamaLLM(model="gemma:2b")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))