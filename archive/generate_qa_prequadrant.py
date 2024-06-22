import requests
import json
from bs4 import BeautifulSoup
import time
import os
from datetime import datetime
import PyPDF2
import io
from Crypto.Cipher import AES

# Load configuration
CONFIG_FILE = 'config.json'
with open(CONFIG_FILE, 'r') as config_file:
    config_list = json.load(config_file)
    config = config_list[0]  # Assuming the config file contains a list with one dictionary

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
HOME_TEAM = config.get('home_team')
INCUMBENT_NAME = config.get('incumbent_name')

# Create results folder if it doesn't exist
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Function to get answer or summary from Ollama server
def get_ollama_response(prompt, model=MODEL):
    headers = {'Content-Type': 'application/json'}
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False  # Ensure the response is not streamed
    }
    try:
        response = requests.post(SERVER_URL, headers=headers, json=payload)
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
        response_data = response.json()
        return response_data.get('response', 'No response found')
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Ollama server: {e}")
        return "Error communicating with Ollama server"

# Function to generate search queries using the language model
def generate_search_queries(prompt, home_team, incumbent_name):
    print("Generating search queries using the LLM...")
    query_prompt = f"Generate search queries for the following prompt. Ensure to include the incumbent ({incumbent_name}) and home team ({home_team}) where relevant:\n\n{prompt}"
    queries_response = get_ollama_response(query_prompt, model=MODEL)
    queries = queries_response.split('\n')
    return [query.strip() for query in queries if query.strip()]

# Summarizer function to filter the web results into a summary
def summarize_and_extract_key_points(web_contents):
    print("Summarizing and extracting key points from web contents...")
    summary = ""
    for content in web_contents:
        if len(content.strip()) > 0:
            prompt = f"Summarize the following content:\n\n{content}"
            summarized_text = get_ollama_response(prompt)
            summary += summarized_text + "\n"
    return summary

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
        if url.lower().endswith('.pdf'):
            return extract_pdf_content(response.content)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([para.get_text() for para in paragraphs])
        return content
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} for URL: {url}")
    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
    return ""

# Function to extract text content from a PDF file
def extract_pdf_content(pdf_content):
    try:
        pdf_text = ""
        pdf_file = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        if pdf_file.is_encrypted:
            pdf_file.decrypt('')
        for page_num in range(len(pdf_file.pages)):
            pdf_text += pdf_file.pages[page_num].extract_text()
        return pdf_text
    except Exception as e:
        print(f"Error extracting PDF content: {e}")
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

def process_web_content(web_links, num_links=NUM_LINKS):
    # Extract content from web links
    print("Extracting content from web links...")
    web_content = []
    used_links = []
    for link in web_links:
        content = extract_content(link)
        if content:
            web_content.append(content)
            used_links.append(link)
        else:
            print(f"Content extraction failed for URL: {link}")
        if len(web_content) >= num_links:
            break
    web_content_text = "\n".join(web_content)

    if len(web_content) < num_links:
        print(f"Warning: Only {len(web_content)} web content entries were extracted, less than the desired {num_links}.")

    # Summarize and extract key points from web content
    cleaned_web_content = summarize_and_extract_key_points(web_content)

    return web_content_text, cleaned_web_content, used_links

# Initialize summary statistics
total_questions = len(questions)
total_web_hits = 0
total_processing_time = 0

# Initial general queries about the incumbent
initial_queries_prompt = f"Generate search queries to understand the incumbent's capabilities, comparative analysis, and specific strengths and weaknesses. Ensure to include the incumbent ({INCUMBENT_NAME}) and home team ({HOME_TEAM})."
initial_queries = generate_search_queries(initial_queries_prompt, HOME_TEAM, INCUMBENT_NAME)

initial_info = []
initial_links = []

for query in initial_queries:
    info, links = web_search(query)
    initial_info.extend(info)
    initial_links.extend(links)

initial_web_content_text, initial_cleaned_web_content, initial_used_links = process_web_content(initial_links)

# Accumulated context
context = base_context + f"\nInitial Web Summary: {initial_cleaned_web_content}\n\n"

# Iterate over questions and get answers, including previous Q&A in the context
answers = {}

for i, question in enumerate(questions):
    start_time = time.time()
    
    print(f"\nProcessing question {i+1}/{len(questions)}: {question}")

    # Extract the tag and actual question
    if "}" in question:
        tag, actual_question = question.split("}", 1)
        tag = tag.strip("{").strip()
        actual_question = actual_question.strip()
    else:
        tag = "general"
        actual_question = question.strip()

    if tag == "incumbent":
        print("Tag identified as 'incumbent'. Generating search queries for the question relative to the incumbent...")
        # Generate queries specific to the question related to the incumbent
        question_queries_prompt = f"Generate search queries for the following question relative to the incumbent ({INCUMBENT_NAME}). Ensure to include the incumbent ({INCUMBENT_NAME}) and home team ({HOME_TEAM}): {actual_question}"
    elif tag == "process":
        print("Tag identified as 'process'. Generating search queries for the question related to process...")
        # Generate queries specific to the question related to process
        question_queries_prompt = f"Generate search queries for the following question related to process. Ensure to include the incumbent ({INCUMBENT_NAME}) and home team ({HOME_TEAM}): {actual_question}"
    else:
        print("Tag not identified. Defaulting to general search queries...")
        # Generate default queries for general questions
        question_queries_prompt = f"Generate search queries for the following question. Ensure to include the incumbent ({INCUMBENT_NAME}) and home team ({HOME_TEAM}): {actual_question}"

    question_queries = generate_search_queries(question_queries_prompt, HOME_TEAM, INCUMBENT_NAME)

    question_info = []
    question_links = []

    for query in question_queries:
        info, links = web_search(query)
        question_info.extend(info)
        question_links.extend(links)

    web_info = initial_info + question_info
    web_links = initial_links + question_links

    web_info_text = "\n".join(web_info)
    web_links_text = "\n".join(web_links)
    num_web_hits = len(web_info)
    total_web_hits += num_web_hits
    
    time.sleep(1)  # Add a small delay for readability in the console output
    print("Web info gathered.")

    web_content_text, cleaned_web_content, used_links = process_web_content(web_links)

    print("Waiting to generate answer...")
    prompt_context = context + f"\nWeb Summary: {cleaned_web_content}\n\n"
    prompt = f"{INPUT_PROMPT}\n\nContext: {prompt_context}\nQuestion: {actual_question}\nGoal: Help the Home Team ({HOME_TEAM}) win the business from the incumbent ({INCUMBENT_NAME})."
    answer = get_ollama_response(prompt, model=MODEL)
    answers[actual_question] = {
        "answer": answer,
        "web_hits": num_web_hits,
        "web_info": web_info_text,
        "web_links": web_links_text,
        "web_content": web_content_text,
        "cleaned_web_content": cleaned_web_content,
        "prompt": prompt,
        "used_links": used_links
    }
    context += f"\nQ: {actual_question}\nA: {answer}\nWeb Info: {web_info_text}\nWeb Links: {web_links_text}\nWeb Content: {web_content_text}\nCleaned Web Content: {cleaned_web_content}\nWeb Hits: {num_web_hits}\n"  # Append the Q&A to the context
    print(f"Answer generated and added to context for question {i+1}\n" + "-" * 50)

    # Save individual results
    with open(result_file, 'a', encoding='utf-8') as f:
        f.write(f"## Q: {actual_question}\n")
        f.write(f"**A:** {answers[actual_question]['answer']}\n")
        f.write(f"**References:**\n")
        for link in used_links:
            f.write(f"- {link}\n")
        f.write("\n")

    with open(web_data_file, 'a', encoding='utf-8') as f:
        f.write(f"## Q: {actual_question}\n")
        f.write(f"**Web Hits:** {answers[actual_question]['web_hits']}\n")
        f.write(f"**Web Info:**\n{answers[actual_question]['web_info']}\n\n")
        f.write(f"**Web Links:**\n{answers[actual_question]['web_links']}\n\n")
        f.write(f"**Web Content:**\n{answers[actual_question]['web_content']}\n\n")
        f.write(f"**Cleaned Web Content:**\n{answers[actual_question]['cleaned_web_content']}\n\n")
        f.write("\n" + "-" * 50 + "\n")

    with open(prompt_file, 'a', encoding='utf-8') as f:
        f.write(f"## Q: {actual_question}\n")
        f.write(f"**Prompt Sent:**\n{answers[actual_question]['prompt']}\n\n")
        f.write("\n" + "-" * 50 + "\n")

    # Calculate processing time for the question
    end_time = time.time()
    processing_time = end_time - start_time
    total_processing_time += processing_time

# Generate final summary with context including incumbent and home team names
final_summary_prompt = (
    f"Summarize the opportunity, the incumbent's strengths and weaknesses, "
    f"the Home Team's strengths and weaknesses, an analysis of the opportunity, "
    f"and whether the Home Team ({HOME_TEAM}) should try and perform the work based on the Q&A results. "
    f"Consider the following context: \n\n"
    f"Incumbent: {INCUMBENT_NAME}\n\n"
    f"Context from previous Q&A results:\n\n"
    f"{context}"
)

final_summary = get_ollama_response(final_summary_prompt, model=MODEL)

# Save summary statistics and configuration variables to the results file
with open(result_file, 'r+', encoding='utf-8') as f:
    content = f.read()
    f.seek(0, 0)
    f.write(f"# Summary Statistics\n")
    f.write(f"- Total Questions: {total_questions}\n")
    f.write(f"- Total Web Hits: {total_web_hits}\n")
    f.write(f"- Total Processing Time: {total_processing_time:.2f} seconds\n\n")
    f.write(f"# Configuration Variables\n")
    f.write(f"- Model: {MODEL}\n")
    f.write(f"- Data Folder: {DATA_FOLDER}\n")
    f.write(f"- Capabilities File: {CAPABILITIES_FILE}\n")
    f.write(f"- SOW File: {SOW_FILE}\n")
    f.write(f"- Questions File: {QUESTIONS_FILE}\n")
    f.write(f"- Results Folder: {RESULTS_FOLDER}\n")
    f.write(f"- Number of Links: {NUM_LINKS}\n")
    f.write(f"- Server URL: {SERVER_URL}\n")
    f.write(f"- Search URL: {SEARCH_URL}\n")
    f.write(f"- Web Search Company Name: {WEB_SEARCH_COMPANY_NAME}\n")
    f.write(f"- Input Prompt: {INPUT_PROMPT}\n")
    f.write(f"- Home Team: {HOME_TEAM}\n")
    f.write(f"- Incumbent Name: {INCUMBENT_NAME}\n\n")
    f.write(f"# BLUF (Bottom Line Up Front) Summary\n")
    f.write(final_summary + "\n\n")
    f.write(content)

print(f"Results saved to {result_folder}")
