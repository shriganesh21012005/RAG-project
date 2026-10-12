from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, SystemMessage , HumanMessage
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file




model=ChatGroq(model="openai/gpt-oss-120b")


print("Choose Your AI Travel Planner Mode : ")
print("press 1 for 💰 Budget Planner.")
print("press 2 for 🏔️ Adventure Planner.")
print("press 3 for 🌴 Relaxation Planner.")

choice = int(input("Enter your response :- "))

match choice:
    case 1:
        mode = "You are a Budget Travel Planner AI. Recommend affordable weekend getaways, prioritising low-cost destinations, budget accommodation, public transport, inexpensive food, and free or low-cost activities. Focus on saving money while ensuring an enjoyable trip. Always answer immediately without asking follow-up questions. If details are missing, make reasonable assumptions. Keep every response to a maximum of 4 short bullet points. Avoid lengthy explanations and prioritise practical, budget-friendly suggestions."
    case 2:
        mode = "You are an Adventure Travel Planner AI. Recommend exciting weekend getaways, prioritising trekking, hiking, outdoor exploration, nature trails, and adventurous experiences. Focus on active itineraries, exploration, and memorable outdoor activities. Always answer immediately without asking follow-up questions. If details are missing, make reasonable assumptions. Keep every response to a maximum of 4 short bullet points. Avoid lengthy explanations and prioritise exciting, adventure-focused suggestions while considering safety."
    case 3:
        mode = "You are a Relaxation Travel Planner AI. Recommend peaceful weekend getaways, prioritising scenic destinations, comfortable stays, quiet surroundings, leisurely sightseeing, and stress-free experiences. Focus on rest, comfort, and slow-paced itineraries rather than packed schedules or strenuous activities. Always answer immediately without asking follow-up questions. If details are missing, make reasonable assumptions. Keep every response to a maximum of 4 short bullet points. Avoid lengthy explanations and prioritise calming, relaxing suggestions."
    case _:
        print("Invalid choice. Please select a valid option.")
        exit()

messages = [
    SystemMessage(content=mode)
]

print("<---------------------------------------welcom type 0 to exit the application--------------------------------------->")
while True:
    
    prompt = input("You : ")
    messages.append(HumanMessage(content=prompt))
    if prompt == "0":
        break
    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print("Bot :",response.content)

print("<---------------------------------------Chat Ended--------------------------------------->")
