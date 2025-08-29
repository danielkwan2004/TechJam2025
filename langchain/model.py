from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model("gemini-2.5-flash-lite", model_provider="google_genai")

