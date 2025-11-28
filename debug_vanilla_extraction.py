from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

# Replicate the initialization from benchmark.py
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0, max_retries=1)

question = "What is the 'doctrine of severability'?"
prompt_fmt = f"""Question: {question}

Provide a legal answer. Then, list relevant citations in a separate section titled "Citations:".
"""

print(f"--- PROMPT ---\n{prompt_fmt}\n")
print("--- INVOKING MODEL ---")
try:
    response = llm.invoke(prompt_fmt)
    content = response.content
    print(f"--- RESPONSE CONTENT REPR ---\n{repr(content)}\n")

    # Replicate extraction logic
    retrieved = []
    if "Citations:" in content:
        citation_section = content.split("Citations:")[-1]
        retrieved = [line.strip().strip('- ') for line in citation_section.split('\n') if line.strip()]
        print(f"--- EXTRACTED CITATIONS ---\n{retrieved}")
    else:
        print("--- EXTRACTION FAILED: 'Citations:' not found in response ---")

except Exception as e:
    print(f"--- ERROR ---\n{e}")
