

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


from langchain.chat_models import init_chat_model

model=init_chat_model("groq:openai/gpt-oss-120b")

print(model) #you can see what model is being used. And what can you send there (all details)


response = model.invoke("Why do parrots talk?")

print(response.text) #you can see the response from the model




#OR




from langchain_groq import ChatGroq

model=ChatGroq(model="openai/gpt-oss-120b")
response = model.invoke("Why do parrots talk?")
print(response.text) #you can see the response from the model
