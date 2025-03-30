from dotenv import load_dotenv

load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",temprature = 1.5,output_token=20)
result = llm.invoke("""youa are a chat bot which creates a json data, i will give you a topic like anything, for example aws , you have to convert that into json data like "{"provider":"aws","description":100words of that topic} and some other features likes pricing etc so here is the topic aws,google, azure""")

print(result.content)