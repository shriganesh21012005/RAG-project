from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_classic.retrievers.multi_query import MultiQueryRetriever

from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)



data = PyPDFLoader("document_loading/xyz.pdf") #Loading Document

docs=data.load() #Creating Document object

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10) # Creating chuck details for our document object
texts=text_splitter.split_documents(docs) # Chunking our document object

vectorstore = Chroma.from_documents(
    documents = texts,
    embedding= embeddings,
    persist_directory= "chroma-db"
)


retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
) #retriver used to retrieve the documents for each queries from the vectorstore

model=ChatGroq(model="openai/gpt-oss-120b") #model created to make multiple queires form ne user query.

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=model
) #multi query retriever created to make multiple queires form ne user query and then retrieve the documents for each queries from the vectorstore

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """{system_message}

            Use ONLY the provided context to answer the question.
            If the answer is not present in the context, say: "I could not find the answer in the document."
            However first understand what the student is asking, if he/she is asking any general questiont like how are you, who are you, what i syour name etc etc then answer accordingly, but whenever he/she is asking any question related to context then answer strictly accordingly to the context given and say "I could not find the answer in the document." for cases where the answer is not found.
            """
        ),
        (
            "human",
            """Previous Conversation:{chat_history}
            Retrieved PDF Context:{context}
            Current Question:{question}"""
        )
    ]
)
print("Choose the mode of your AI Document Assistant : ")

while True:
    
    print("press 1 for 😊 Friendly Mode.")
    print("press 2 for 📚 Exam Mode.")
    print("press 3 for 🔥 Strict Mode.")
  
    prompt_choice = int(input(f"\nEnter your response :- "))

    match prompt_choice:
        case 1:
            mode = "You are a friendly, patient, and supportive AI tutor. Make the student feel comfortable asking questions, even if they lose focus or need repeated explanations. Start naturally with reassuring phrases such as `No worries! Let's understand this step by step.` Explain concepts in clear, beginner-friendly language, breaking difficult ideas into manageable parts. Use a warm, conversational tone and guide the student patiently without making them feel judged. Adapt your explanation when the student says they do not understand. Use only the provided PDF context."
            break
        case 2:
            mode = "You are an exam-revision tutor assisting a student who has only one hour left before an exam. Be highly focused, concise, and exam-oriented. Start with phrases such as `Time is limited. Focus on these key points.` Present answers using numbered points, essential definitions, important keywords, and formulas where relevant. Highlight what the student should memorize and finish with a clearly labeled `LAST-MINUTE REVISION` section containing the most important takeaways. Avoid lengthy introductions and unnecessary explanations. Prioritize information from the provided PDF. Never invent missing information; state when the answer cannot be found in the document."
            break
        case 3:
            mode = "You are a strict, disciplined, and results-oriented tutor. Maintain a firm, direct, no-nonsense tone. Start with strict phrases example: `Focus and be an attentive student !`, `ok so Stop being lazy and focus on this topic now` Be direct about misunderstandings and insist on conceptual clarity rather than superficial memorization. Explain the material in a structured manner, identify the central concept, and point out what the student must understand before moving forward. Avoid excessive reassurance, casual conversation, and unnecessary filler. Be demanding about learning, but never insult, humiliate, or belittle the student. Use only the provided PDF context and clearly state when the answer is unavailable."
            break
        case _:
            print("Invalid choice. Please select a valid option.")

chat_history=[] #list to store our converstaions

while True:
    query =input(f"\nYOU : ").strip() #user query
    if query == "0":
        break
    

    docs = multi_query_retriever.invoke(query) # getting the final context form the vector store.

    unified_context = "\n\n".join(
        [doc.page_content for doc in docs]
    )


    ready_prompt = prompt.invoke({
        "context" :unified_context,
        "question": query,
        "system_message": mode,
        "chat_history": chat_history
    }) # Getting the final peompt to send to LLM

    LLM_response = model.invoke(ready_prompt) #getting the final LLM repsonse
    chat_history.append(query) #storing the user query in memory
    chat_history.append(LLM_response.content) #storing the bot response in chat history
    print(f"\nBot : {LLM_response.content}")

print("<---------------------------------------Chat Ended--------------------------------------->")
# print(chat_history)
