import json
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Annotated

# Load environment variables (if needed)
load_dotenv()

# Initialize the model
model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

# Define schema using Pydantic
class OutputDataSchema(BaseModel):
    FullNameOfTheStudent: Annotated[str, Field(description="Full name of the student")]
    Maths: Annotated[float, Field(description="Maths subject marks")]
    Science: Annotated[float, Field(description="Science subject marks")]
    English: Annotated[float, Field(description="English subject marks")]
    Computer: Annotated[float, Field(description="Computer subject marks")]
    overall_Percentage: Annotated[float, Field(description="Total percentage")]
    Schoolname: Annotated[str, Field(description="Name of the school")]

# Prompt Template
template = PromptTemplate.from_template("""
Extract the student's details and return a structured JSON format.

                    
Example:
Student Description:
"Rahul Sharma performed well this semester. He scored 90 in Mathematics, 85 in Science, 80 in English, and 95 in Computer Science."



Now extract details from the following text:
{student_detail}    
""")

# Streamlit UI
st.header("Student Performance Extractor")
student_detail = st.text_input("Enter student details with all necessary information")

prompt = template.invoke({'student_detail': student_detail})
st_model = model.with_structured_output(OutputDataSchema)

if st.button("Get JSON Data"):
    response_obj = st_model.invoke(prompt)  # This returns a Pydantic object, not a dict

    # Convert Pydantic object to JSON
    if isinstance(response_obj, OutputDataSchema):
        response_json = response_obj.model_dump()  # Use model_dump() for conversion
        st.json(response_json)  # Display structured JSON
    else:
        st.error("Unexpected response format. Expected a JSON-compatible object.")
