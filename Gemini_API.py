'''''''''
Explanation:
Configuration: You set the API key using genai.configure(...). To get an API key go to https://ai.google.dev/gemini-api/docs/api-key.
Select a Model: “gemini-2.0-flash-exp” is provided as the generative model.
User Prompt: A single question (“What is GenAI?”) is passed to the model.
Response: The model returns generated text based on its trained knowledge.
Exercise:
Change the prompt to something else, like “Explain the basics of data structures in Python.”
Modify the model name to another version to see if you get different responses.
'''

import os
from dotenv import load_dotenv
import google.generativeai as genai

# Configure the client with your API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Specify the model to use
# model = genai.GenerativeModel("gemini-2.0-flash-exp")
model = genai.GenerativeModel("gemini-1.5-flash")

# Prompt the user for input
user_prompt = "What is FM synthesis? Please, answer only in 5 bulletpoints."

# Generate a response using the Gemini API
response = model.generate_content(user_prompt)

print("Generated text:\n", response.text)


'''  gemini-2.0-flash-exp
*   **Frequency Modulation:** At its core, FM synthesis uses one waveform (the modulator) to modulate the frequency of another waveform (the carrier).
*   **Complex Harmonics:** This modulation creates complex harmonic spectra, often resulting in bright, metallic, and evolving sounds.
*   **Operator-Based:** Synthesizers typically implement FM with interconnected "operators," each acting as both a modulator and/or carrier.
*   **Algorithms:** The way these operators are connected (the "algorithm") determines the final sound character. Changing the algorithm dramatically alters the sound.
*   **Beyond Sine Waves:** While sine waves are common, FM can use other waveforms (square, triangle, saw) for even richer timbral possibilities.

     gemini-1.5-flash
* Uses one oscillator (modulator) to change the characteristics of another (carrier).
* Modulator's output changes the carrier's frequency (or other parameters).
* Creates complex, rich timbres from simpler waveforms.
* Frequency modulation depth controls the timbre's complexity.
* Widely used in synthesizers and sound design.

'''