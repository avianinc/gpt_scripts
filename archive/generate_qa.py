from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import logging
import os
import requests
from tqdm.auto import tqdm
import torch
from bs4 import BeautifulSoup
import time

# Set up logging to be more verbose
logging.set_verbosity_info()

# Directories
model_directory = './models/falcon-7b'
tokenizer_directory = model_directory

# Ensure the model directory exists
os.makedirs(model_directory, exist_ok=True)

# Hugging Face repository for Falcon 7B quantized
hf_model_name = "tiiuae/falcon-7b"

# Function to download a file with a progress bar
def download_file(url, destination):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    t = tqdm(total=total_size, unit='iB', unit_scale=True, desc=destination)
    with open(destination, 'wb') as file:
        for data in response.iter_content(block_size):
            t.update(len(data))
            file.write(data)
    t.close()

# Download and save the model and tokenizer
print("Downloading and loading model and tokenizer from Hugging Face...")
torch.cuda.empty_cache()  # Clear GPU memory

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)  # Load in 4-bit with correct compute dtype

# Load the Falcon model and tokenizer with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(hf_model_name, quantization_config=bnb_config, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
print("Model and tokenizer loaded successfully from Hugging Face.")

# Save the model and tokenizer locally
model.save_pretrained(model_directory)
tokenizer.save_pretrained(tokenizer_directory)
print(f"Model and tokenizer saved to {model_directory}")

# The model is already set to the correct device by `accelerate`, so we don't need to move it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the SOW text from a file
print("Loading SOW text...")
with open('sow.txt', 'r', encoding='utf-8') as sow_file:
    sow_text = sow_file.read()
print("SOW text loaded.")

# Load the company's capabilities from a file
print("Loading company's capabilities...")
with open('capabilities.txt', 'r', encoding='utf-8') as capabilities_file:
    capabilities_text = capabilities_file.read()
print("Company's capabilities loaded.")

# Load the questions from a file
print("Loading questions from file...")
with open('questions.txt', 'r', encoding='utf-8') as questions_file:
    questions = [line.strip() for line in questions_file.readlines()]
print("Questions loaded.")

# Combine SOW and capabilities into a single context
base_context = f"SOW: {sow_text}\n\nCapabilities: {capabilities_text}\n\n"

# Function to perform web search and print results
def web_search(query):
    print(f"Performing web search for query: {query}")
    search_url = f"https://www.google.com/search?q={query}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    results = []
    result_links = []
    for g in soup.find_all(class_='BNeawe vvjwJb AP7Wnd'):
        result_title = g.get_text()
        results.append(result_title)
    for link in soup.find_all('a'):
        href = link.get('href')
        if "url?q=" in href and not "webcache" in href:
            result_link = href.split("?q=")[1].split("&sa=U")[0]
            result_links.append(result_link)
    return results, result_links

# Function to chunk input text into smaller segments
def chunk_text(text, max_length):
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    return chunks

# Function to get answer for a question
def get_answer(context, question, web_info, web_links):
    print(f"Generating answer for question: {question}")
    input_text = f"{context}\nWeb Info: {web_info}\n\nQuestion: {question}\nWeb Links: {web_links}\n"
    input_ids_chunks = chunk_text(input_text, 4096)  # Chunking input text to fit max length

    # Generate answers for each chunk
    answers = []
    for input_ids in input_ids_chunks:
        input_ids = torch.tensor([input_ids]).to(device)
        attention_mask = torch.ones(input_ids.shape).to(device)

        outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=200, num_return_sequences=1, do_sample=True, top_k=50, top_p=0.95, temperature=1.0)  # Adjust parameters for better streaming
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        answers.append(generated_text)
    return " ".join(answers)

# Iterate over questions and get answers, including previous Q&A in the context
answers = {}
context = base_context  # Start with the base context

for i, question in enumerate(questions):
    print(f"\nProcessing question {i+1}/{len(questions)}: {question}")
    print("Waiting for web search results...")
    web_info, web_links = web_search(f"Draken International {question}")
    web_info_text = "\n".join(web_info)
    web_links_text = "\n".join(web_links)
    time.sleep(1)  # Add a small delay for readability in the console output
    print("Web info gathered.")
    print("Waiting to generate answer...")
    answer = get_answer(context, question, web_info_text, web_links_text)
    answers[question] = answer
    context += f"\nQ: {question}\nA: {answer}\n"  # Append the Q&A to the context
    print(f"Answer generated and added to context for question {i+1}\n" + "-" * 50)

# Optional: Save the Q&A to a file
print("Saving results to QandA_results.txt...")
with open("QandA_results.txt", 'w', encoding='utf-8') as f:
    for question, answer in answers.items():
        f.write(f"Q: {question}\n")
        f.write(f"A: {answer}\n")
        f.write("-" * 50 + "\n")
print("Results saved to QandA_results.txt")
