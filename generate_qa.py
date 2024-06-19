import requests
import json
from bs4 import BeautifulSoup
import time
import os
from datetime import datetime

# Load configuration
CONFIG_FILE = 'config.json'
with open(CONFIG_FILE, 'r') as config_file:
    config = json.load(config_file)

MODEL = config.get('model')
DATA_FOLDER = config.get('data_folder')
CAPABILITIES_FILE = config.get('capabilities_file')
SOW_FILE = config.get('sow_file')
QUESTIONS_FILE = config.get('questions_file')
RESULTS_FOLDER = config.get('results_folder', 'results')
NUM_LINKS = config.get('num_links', 5)  # Default to 5 if not specified
SERVER_URL = config.get('server_url')
SEARCH_URL = config.get('search_url')
WEB_SEARCH_COMPANY_NAME = config.get('web_search_company_name')
INPUT_PROMPT = config.get('input_prompt')

# Create results folder if it doesn't exist
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Function to get answer from Ollama server
def get_answer(question, context, model=MODEL):
    headers = {'Content-Type': 'application/json'}
    payload = {
        "model": model,
        "prompt": f"{INPUT_PROMPT}\n\nContext: {context}\nQuestion: {question}",
        "stream": False  # Ensure the response is not streamed
    }
    try:
        response = requests.post(SERVER_URL, headers=headers, json=payload)
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
        answer = response.json().get('response', 'No answer found')
        return answer, payload['prompt']
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Ollama server: {e}")
        return "Error communicating with Ollama server", payload['prompt']

# Function to perform web search and get top results
def web_search(query):
    print(f"Performing web search for query: {query}")
    search_query = query.replace(" ", "+")
    search_url_full = f"{SEARCH_URL}?q={search_query}&format=json"
    print(f"Web search URL: {search_url_full}")
    try:
        response = requests.get(search_url_full)
        response.raise_for_status()
        search_results = response.json()
        
        results = []
        result_links = []
        for result in search_results.get('results', []):
            results.append(result.get('title'))
            result_links.append(result.get('url'))
        
        num_web_hits = len(results)
        print(f"Found {num_web_hits} web hits.")
        for i, title in enumerate(results[:5]):
            print(f"Result {i+1}: {title} - {result_links[i]}")
        return results, result_links
    except Exception as e:
        print(f"Error during web search: {e}")
        return [], []

# Function to extract content from a URL with error handling and retries
def extract_content(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([para.get_text() for para in paragraphs])
        return content
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} for URL: {url}")
    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
    return ""

# Load the SOW text from a file
print("Loading SOW text...")
with open(f'{DATA_FOLDER}/{SOW_FILE}', 'r', encoding='utf-8') as sow_file:
    sow_text = sow_file.read()
print("SOW text loaded.")

# Load the company's capabilities from a file
print("Loading company's capabilities...")
with open(f'{DATA_FOLDER}/{CAPABILITIES_FILE}', 'r', encoding='utf-8') as capabilities_file:
    capabilities_text = capabilities_file.read()
print("Company's capabilities loaded.")

# Load the questions from a file
print("Loading questions from file...")
with open(f'{DATA_FOLDER}/{QUESTIONS_FILE}', 'r', encoding='utf-8') as questions_file:
    questions = [line.strip() for line in questions_file.readlines()]
print("Questions loaded.")

# Combine SOW and capabilities into a single context
base_context = f"SOW: {sow_text}\n\nCapabilities: {capabilities_text}\n\n"

# Timestamp for filenames and folder
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
result_folder = f"{RESULTS_FOLDER}/{timestamp}"
os.makedirs(result_folder, exist_ok=True)

# Files for results, web data, and prompts
result_file = f"{result_folder}/QandA_results_{timestamp}.md"
web_data_file = f"{result_folder}/Web_data_{timestamp}.md"
prompt_file = f"{result_folder}/Prompts_{timestamp}.md"

# Initialize files
with open(result_file, 'w', encoding='utf-8') as f:
    f.write(f"# Q&A Results ({timestamp})\n")
with open(web_data_file, 'w', encoding='utf-8') as f:
    f.write(f"# Web Data ({timestamp})\n")
with open(prompt_file, 'w', encoding='utf-8') as f:
    f.write(f"# Prompts ({timestamp})\n")

# Iterate over questions and get answers, including previous Q&A in the context
answers = {}
context = base_context  # Start with the base context

for i, question in enumerate(questions):
    print(f"\nProcessing question {i+1}/{len(questions)}: {question}")
    print("Waiting for web search results...")
    web_info, web_links = web_search(f"{WEB_SEARCH_COMPANY_NAME} {question}")
    web_info_text = "\n".join(web_info)
    web_links_text = "\n".join(web_links)
    num_web_hits = len(web_info)
    time.sleep(1)  # Add a small delay for readability in the console output
    print("Web info gathered.")

    print("Extracting content from web links...")
    web_content = []
    for link in web_links:
        content = extract_content(link)
        if content:
            web_content.append(content)
        else:
            print(f"Content extraction failed for URL: {link}")
        if len(web_content) >= NUM_LINKS:
            break
    web_content_text = "\n".join(web_content)

    if len(web_content) < NUM_LINKS:
        print(f"Warning: Only {len(web_content)} web content entries were extracted, less than the desired {NUM_LINKS}.")

    print("Waiting to generate answer...")
    answer, prompt_sent = get_answer(question, context, model=MODEL)
    answers[question] = {
        "answer": answer,
        "web_hits": num_web_hits,
        "web_info": web_info_text,
        "web_links": web_links_text,
        "web_content": web_content_text,
        "prompt": prompt_sent
    }
    context += f"\nQ: {question}\nA: {answer}\nWeb Info: {web_info_text}\nWeb Links: {web_links_text}\nWeb Content: {web_content_text}\nWeb Hits: {num_web_hits}\n"  # Append the Q&A to the context
    print(f"Answer generated and added to context for question {i+1}\n" + "-" * 50)

    # Save individual results
    with open(result_file, 'a', encoding='utf-8') as f:
        f.write(f"## Q: {question}\n")
        f.write(f"**A:** {answers[question]['answer']}\n\n")

    with open(web_data_file, 'a', encoding='utf-8') as f:
        f.write(f"## Q: {question}\n")
        f.write(f"**Web Hits:** {answers[question]['web_hits']}\n")
        f.write(f"**Web Info:**\n{answers[question]['web_info']}\n\n")
        f.write(f"**Web Links:**\n{answers[question]['web_links']}\n\n")
        f.write(f"**Web Content:**\n{answers[question]['web_content']}\n\n")
        f.write("\n" + "-" * 50 + "\n")

    with open(prompt_file, 'a', encoding='utf-8') as f:
        f.write(f"## Q: {question}\n")
        f.write(f"**Prompt Sent:**\n{answers[question]['prompt']}\n\n")
        f.write("\n" + "-" * 50 + "\n")

print(f"Results saved to {result_folder}")
