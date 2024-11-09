from flask import Flask, render_template, request, jsonify
from transformers import BartTokenizer, BartForConditionalGeneration, AutoTokenizer, AutoModelForQuestionAnswering, DPRContextEncoder, DPRQuestionEncoder
import torch

# Initialize Flask app
app = Flask(__name__)

# Load BART model and tokenizer
bart_tokenizer = BartTokenizer.from_pretrained("./fine_tuned_bart2")  
bart_model = BartForConditionalGeneration.from_pretrained("./fine_tuned_bart2")

# Load Question Answering model and tokenizer
qa_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
qa_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")

# Load DPR question encoder and context encoder
dpr_question_tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
dpr_question_model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
dpr_context_tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
dpr_context_model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

def generate_response(question, max_length=256):
    # Step 1: Use the DPR model to encode the question (to simulate retrieval)
    question_inputs = dpr_question_tokenizer(question, return_tensors="pt", padding=True, truncation=True)
    question_embedding = dpr_question_model(**question_inputs).pooler_output

    # Step 2: Generate a response using BART (using the question as context)
    bart_inputs = bart_tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    bart_outputs = bart_model.generate(
        bart_inputs['input_ids'],
        max_length=max_length,  
        num_return_sequences=1,
        temperature=0.5,
        top_p=0.85,
        do_sample=True,
        top_k=50
    )
    bart_response = bart_tokenizer.decode(bart_outputs[0], skip_special_tokens=True)

    # Step 3: Use the BART response as input to the QA model to get the final answer
    qa_inputs = qa_tokenizer.encode_plus(
        bart_response, 
        question, 
        return_tensors='pt',
        add_special_tokens=True
    )

    with torch.no_grad():
        outputs = qa_model(**qa_inputs)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits)

    # Get the refined answer from the model's output
    answer = qa_tokenizer.convert_tokens_to_string(qa_tokenizer.convert_ids_to_tokens(qa_inputs['input_ids'][0][answer_start: answer_end + 1]))

    # Combine both the BART and QA model responses into one response (either BART or QA)
    if answer.strip():  # If QA model provides a valid answer
        final_response = answer
    else:  # If QA model doesn't provide a good answer, fall back to BART's response
        final_response = bart_response

    return final_response

# Route for the home page
@app.route("/")
def home():
    return render_template("index.html")  # Renders index.html from templates folder

# Route for handling chatbot responses
@app.route("/get_response", methods=["POST"])
def get_response():
    # Get user input from the POST request (sent in JSON format)
    user_input = request.json.get("question")
    
    if user_input:
        # Generate the response using the BART and QA models
        response = generate_response(user_input)
        return jsonify({"response": response})
    
    return jsonify({"response": "Sorry, I didn't understand that. Please try again!"})

if __name__ == "__main__":
    # Run the Flask app in debug mode for development
    app.run(debug=True)
