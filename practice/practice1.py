from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
import streamlit as st
load_dotenv()

# invoking model
model = GoogleGenerativeAI(model='gemini-1.5-flash')
response = model.invoke("hello how are you")
print(response)


# using prompt template
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
history = [
    SystemMessage(content='You are a user chat histort=y manager')
]

template = PromptTemplate.from_template("""
you are a chat assistant. how to convert the {text} in {language} language and if you are unable to do it then simply reponse "sorry i am  unable to convert "
""")

text = st.text_input("enter text in english")
language  = st.selectbox(
    'Select your langugae',
 ('hindi', 'french', 'gujarati'))
prompt = template.invoke({'text':text,'language':language})

if st.button("Convert"):
    res_hindi = model.invoke(prompt)
    temp_dict={
        "user_input":HumanMessage(content=text),
        "ai_response":AIMessage(content=res_hindi)
    }
    history.append(temp_dict)
    st.write(res_hindi)

print(history)