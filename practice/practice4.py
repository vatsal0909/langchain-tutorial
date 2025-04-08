from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
import streamlit as st
from typing import Annotated
from pydantic import BaseModel, Field

load_dotenv()

# Create Output Data Schema for structured output
class OutputDataSchema(BaseModel):
    topic_name: Annotated[str, Field(description="Topic name")]
    hindi_text: Annotated[str, Field(description="Hindi summary of the topic")]
    english_text: Annotated[str, Field(description="English summary of the topic")]
    gujarati_text: Annotated[str, Field(description="gujarati summary of the topic")]

# Initialize the model and parser
parser = StrOutputParser()
model_1 = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
model_2 = ChatGoogleGenerativeAI(model='gemini-1.5-pro')

# Define the prompt templates
prompt_template_1 = PromptTemplate(
    template="You are a summary generator which will generate a scientific summary for the following topic, you have to create a summary of at max 3 paragraphs. Do this for the topic -> {topic}",
    input_variables=['topic']
)

prompt_template_2 = PromptTemplate(
    template="You are a summary generator which will generate a scientific summary for the following topic, you have to create a summary of at max 3 paragraphs. Do this for the topic -> {topic}",
    input_variables=['topic']
)

prompt_template_3 = PromptTemplate(
    template="You have to generate 5 key points to remember in Hindi language, in short, all the important points for the given summary -> {output_summary}",
    input_variables=['output_summary']
)

prompt_template_4 = PromptTemplate(
    template="You have to generate 5 key points to remember in Gujarati language, in short, all the important points for the given summary -> {output_summary}",
    input_variables=['output_summary']
)

prompt_template_5 = PromptTemplate(
    template="You have to merge two outputs and represent them in a beautiful way. Output1 is {output_hindi}, and Output2 is {output_gujarati}. Also, add the English version for the same.",
    input_variables=['output_hindi', 'output_gujarati']
)

# Parallel chain for handling Hindi and Gujarati output
parralel_chain = RunnableParallel({
    'output_hindi': prompt_template_1 | model_1 | parser | prompt_template_3 | model_1 | parser,
    'output_gujarati': prompt_template_2 | model_2 | parser | prompt_template_4 | model_2 | parser
})

# Chain for merging outputs
merging_chain = prompt_template_5 | model_1 | parser

# Initialize Streamlit UI
st.title("Notes Generator in Multiple Languages")

# Input for topic
topic_input = st.text_input("Enter the topic")

# Template for displaying the chain output as JSON
prompt_template_json = PromptTemplate(
    template="{chain_output}",
    input_variables=['chain_output']
)

# Submit button functionality
if st.button("Submit"):
    # Invoke parallel and merging chains
    chain = parralel_chain | merging_chain
    res = chain.invoke({'topic': topic_input})

    # Ensure result is structured as OutputDataSchema
    json_chain = prompt_template_json | model_1.with_structured_output(OutputDataSchema)
    json_data = json_chain.invoke({'chain_output': res})

    # Display the results
    st.write("Generated Notes:")
    st.write(res)

    # Display the structured output as JSON
    st.write("Structured Output (JSON):")
    st.json(json_data.dict())
        
