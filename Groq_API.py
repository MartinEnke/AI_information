''''''
'''
Explanation:
System Prompt: This sets the overall context and tone. Here, we instruct the model to be a “helpful assistant.”
User Prompt: The question or request from the user, e.g., “What is GenAI?”
API Call: combine both prompts in a list of messages and send them to the model for a chat completion.
'''


from groq import Groq
import os
from dotenv import load_dotenv


# Configure the client with your API key
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Initialize the Groq client
client = Groq(api_key=api_key)

# Specify the model to use
model = "llama-3.3-70b-versatile"

# System's task
system_prompt = "You are a helpful assistant."

# User's request
user_prompt = "What is GenAI?"

# Generate a response using the Groq API
response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
)

# Display the generated text
print("Generated text:\n", response.choices[0].message.content)
