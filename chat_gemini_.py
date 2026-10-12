from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

from langchain_google_genai import ChatGoogleGenerativeAI

model=ChatGoogleGenerativeAI(model="gemini-3.7-flash",temperature=0.3,max_tokens=1024) #temperature is a parameter which has a range of (0-1)..and 0.1 will give least creative response and 0.9 will give most creative response. You can set it as per your requirement.

print(model) #you can see what model is being used. And what can you send there (all details)

response = model.invoke("Who are you ?")
print(response.text) #you can see the response from the model



#OR



from langchain.chat_models import init_chat_model
model = init_chat_model("google_genai:gemini-3.7-flash")

response = model.invoke("Why do parrots talk?")
print(response.text) #you can see the response from the model