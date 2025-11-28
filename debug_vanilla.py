from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)

question = "What is the 'doctrine of severability'?"
prompt = f"""Question: {question}

Provide a legal answer. Then, list relevant citations in a separate section titled "Citations:".
"""

print(f"Prompt:\n{prompt}\n")
response = llm.invoke(prompt)
print(f"Response:\n{response.content}\n")

content = response.content
retrieved = []
if "Citations:" in content:
    citation_section = content.split("Citations:")[-1]
    retrieved = [line.strip().strip('- ') for line in citation_section.split('\n') if line.strip()]

print(f"Extracted Citations: {retrieved}")
