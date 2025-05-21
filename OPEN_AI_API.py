'''''''''
Explanation:
Conversation History: The messages array provides the full context of the dialogue, 
including a mistaken assistant response.
Developer/System Role: An optional role message (e.g., “developer”) can set the overarching 
instructions or constraints on the assistant’s behavior.
Correction or Follow-up: The user can question or challenge the assistant’s previous statement. 
The model will use prior messages to generate a coherent response in context.

Exercise:
Change the previous assistant message to something else incorrect or humorous, and see how the model addresses it.
Replace the developer or system prompt with something more specific, e.g., “You have expertise in biology. Correct any scientific misconceptions.”
Vary the temperature or max_tokens to see if it changes how the model revises its previous statement.
'''
import openai

client = openai.OpenAI(api_key="YOUR_API_KEY")

# Specify the model to use
model = "gpt-4o-mini"

# Prompt the user for input
user_prompt = "What is GenAI?"

# Generate a response using the OpenAI API
response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "developer", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": "GenAI is a vegetable!"},
        {"role": "user", "content": "Are you sure? What made you think of that?"}
    ],
    temperature=0.7,
    max_tokens=150
)

# Display the generated text
print("Generated text:\n", response.choices[0].message.content)
