from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langserve import add_routes
from dotenv import load_dotenv
load_dotenv()

#loading chat groq model
groq_api_key=os.getenv('GROQ_API_KEY')
model=ChatGroq(model="llama-3.3-70b-versatile",groq_api_key=groq_api_key)

#Create prompt template
system_template="Translate the following to {language}:"
prompt_template=ChatPromptTemplate.from_messages(
    [
        ("system",system_template),
        ("user",'{text}')
    ]
)

parser=StrOutputParser()

#create chain
chain=prompt_template|model|parser


#App definition
app=FastAPI(
    title="Langchain server",
    version="1.0",
    description="A simple API server using Langchain runnable interfaces"
)

#Adding chain Routes
add_routes(
    app,
    chain,
    path="/chain"
)


#Main - to start Application
#To execute this fastapi application, go to cmd, to the following directory for this file, and run >>>python serve.py
if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="localhost",port=8000)
