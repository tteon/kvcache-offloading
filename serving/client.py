import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# vLLM API Configuration
# Use the key from .env or default to EMPTY if not set (but warn/log if needed)
openai_api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
openai_api_base = "http://localhost:8000/v1"

def main():
    print(f"Connecting to vLLM at {openai_api_base}...")
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    try:
        # Get loaded model
        models = client.models.list()
        model = models.data[0].id
        print(f"Running inference with model: {model}")
    except Exception as e:
        print(f"Error connecting to server or retrieving models: {e}")
        return

    # --- Round 1 ---
    print("\n=== Round 1 ===")
    messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]
    
    try:
        response = client.chat.completions.create(
            model=model, 
            messages=messages,
            temperature=0.7
        )
        content = response.choices[0].message.content
        print("Content for Round 1:\n", content)
    except Exception as e:
        print(f"Error in Round 1: {e}")
        return

    # --- Round 2 ---
    print("\n=== Round 2 ===")
    # Multi-turn Chat
    messages.append({"role": "assistant", "content": content})
    messages.append(
        {
            "role": "user",
            "content": "How many Rs are there in the word 'strawberry'?",
        }
    )
    
    try:
        response = client.chat.completions.create(
            model=model, 
            messages=messages,
            temperature=0.7
        )
        content = response.choices[0].message.content
        print("Content for Round 2:\n", content)
    except Exception as e:
        print(f"Error in Round 2: {e}")

if __name__ == "__main__":
    main()
