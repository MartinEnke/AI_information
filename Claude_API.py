'''''''''''

NOT FOR FREE

Explanation:
temperature: A higher value (closer to 1.0) makes outputs more creative or “random.” 
A lower value (closer to 0.0) makes outputs more deterministic and focused.
max_tokens: Limits how many tokens the model can generate in a single response, controlling response length.
System Message vs. User Message: The system message sets the general style or behavior, 
and the user message is the user’s query or input.

Exercise:
Lower the temperature to 0.3 and observe how the answer changes in style or creativity.
Increase max_tokens to 300 to see if you get a longer response.
Adjust the system message to see how it influences the generated content’s tone or style.
'''

import anthropic
import os
from dotenv import load_dotenv


# Configure the client with your API key
load_dotenv()
api_key = os.getenv("CLAUDE_API_KEY")


# Initialize the Anthropic client
client = anthropic.Anthropic(api_key=api_key)

# Specify the model to use
model = "claude-3-5-sonnet-latest"

# Prompt the user for input
user_prompt = "What means LUFS in sound engineering?"

# Define the system message to set the behavior of the assistant
system_message = "You are a helpful assistant. Provide concise and relevant responses."

# Create the message payload
messages = [
    {"role": "user", "content": user_prompt}
]

# Generate a response using the Claude API
response = client.messages.create(
    model=model,
    system=system_message,
    messages=messages,
    max_tokens=150,
    temperature=0.7
)

# Display the generated text
print("Generated text:\n", response.content[0].text)