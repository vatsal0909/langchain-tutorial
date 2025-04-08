from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
parser = JsonOutputParser()


model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

