from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
load_dotenv()

prompt_template = PromptTemplate.from_template("""you are a summery generator , you have to summerize on topic {topic} and generate paragraps of {paragraph_size} and you have to generate on basis of {paragraph_temprature} """)

prompt_template.save("template.json")