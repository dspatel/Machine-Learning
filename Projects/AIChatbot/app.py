# Flask app to host the AI chatbot
from flask import Flask
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# Initialize Flask app and enable CORS


model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name) # This loads the model and returns a dictionary of the model
tokenizer = AutoTokenizer.from_pretrained(model_name)

conversation_history = []

from flask import request, jsonify
import json




app = Flask(__name__)
CORS(app)
@app.route('/chatbot', methods=['POST'])
def handle_prompt():
    # Get the user input from the request
    data = request.get_json()
    user_input = data.get('input', '')
    
    if user_input.lower() == "exit":
        return jsonify({"response": "Goodbye!"})
    
    # Append the user input to the conversation history
    conversation_history.append(user_input)
    
    # Create a string of the conversation history
    history_string = "\n".join(conversation_history) 
    print
    # Tokenize the input text and conversation history
    inputs = tokenizer.encode(history_string, user_input, return_tensors="pt")

    # Generate a response from the model
    outputs = model.generate(inputs, max_length = 60)
    
    # Decode the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Append the model response to the conversation history
    conversation_history.append(response)

    return response



if __name__ == '__main__':
    app.run()