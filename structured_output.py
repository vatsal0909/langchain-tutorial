#llm talking with apis,database
# working with processing llms output

# generally when we talk with llm it gives us unstructured output, like text. text is a unstructured format of output
# output format like json or any is structured data
#usecases
# creating key informnation from any large data and making it into json data to use as api and store it in database
# use to build agents


#1 working with alreadt llm which provide structured output

# using type dictionary
import json
import streamlit as st
from typing import TypedDict, Annotated, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

st.header("review analyzer")

review = st.text_input("write down review")
template = PromptTemplate.from_template(""" {review}""")
prompt = template.invoke({'review':review})

#defining schema
class modelSchema(TypedDict):
    
    summary:Annotated[str,"A brief summary of 1 lines about major functionality"]
    sentiment:str
    pros:Annotated[Optional[list[str]],"write down all the pros"]
    cons:Annotated[Optional[list[str]],"write down all the cons"]
    mobile_name : Annotated[str,"name of the mobie phone"]

structured_format = model.with_structured_output(modelSchema)
response = structured_format.invoke(prompt)
if st.button("analyze"):
    st.write(response)