from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()
from langchain_core.runnables import RunnableLambda, RunnableParallel,RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def fun_runnable(input:str):
    return print(input),len(input.split())

prompt_template1 = PromptTemplate(
    template="generate brief overview of the follwing topic {topic}",
    input_variables=['topic']
)

prompt_template2 = PromptTemplate(
    template="generate 5 points on {summary}",
    input_variables=['summary']
)

parser = StrOutputParser()

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

chain1 = prompt_template1|model|parser

parallel_chain = RunnableParallel({
    'saved_response': RunnablePassthrough(),
    'key_points':prompt_template2|model|parser
})

chain = chain1|parallel_chain|RunnableLambda(lambda x: fun_runnable(x['key_points']))

print(chain.invoke({'topic':'black hole'}))