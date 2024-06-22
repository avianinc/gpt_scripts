import requests
import json
from bs4 import BeautifulSoup
import time
import os
from datetime import datetime
import PyPDF2
import io
from qdrant_client import QdrantClient
from transformers import BertTokenizer, BertModel
import uuid

# Load configuration
CONFIG_FILE = 'config.json'
with open(CONFIG_FILE, 'r') as config_file:
    config_list = json.load(config_file)
    config = config_list[0]

MODEL = config['model']
DATA_FOLDER = config['data_folder']
CAPABILITIES_FILE = config['capabilities_file']
SOW_FILE = config['sow_file']
QUESTIONS_FILE = config['questions_file']
RESULTS_FOLDER = config.get('results_folder', 'results')
NUM_LINKS = config.get('num_links', 5)
SERVER_URL = config['server_url']
SEARCH_URL = config['search_url']
WEB_SEARCH_COMPANY_NAME = config['web_search_company_name']
INPUT_PROMPT = config['input_prompt']
HOME_TEAM = config['home_team']
INCUMBENT_NAME = config['incumbent_name']
WEB_SEARCH_KEYWORDS = config.get('web_search_keywords', [])

clear_collection = config.get('clear_collection', False)

# Initialize Qdrant client
client = QdrantClient("localhost", port=6333)

# Create collection if it doesn't exist
if clear_collection:
    client.delete_collection("documents")
client.create_collection(
    collection_name="documents",
    vectors_config={
        "size": 768,  # Correct size of the embedding vectors
        "distance": "Cosine"
    }
)

# Initialize tokenizer and embedding model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
embedding_model = BertModel.from_pretrained("bert-base-uncased")

def chunk_text(text, max_length=512):
    tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"].squeeze().tolist()
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

def add_document_to_vector_store(client, collection_name, doc_id, text):
    chunks = chunk_text(text)
    for idx, chunk in enumerate(chunks):
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
        outputs = embedding_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()

        point = {
            "id": str(uuid.uuid4()),  # Use UUID for point ID
            "vector": embedding.tolist(),
            "payload": {"text": chunk}
        }

        try:
            client.upsert(
                collection_name=collection_name,
                points=[point]
            )
        except Exception as e:
            print(f"Error upserting point {point['id']}: {e}")

def generate_search_queries(prompt, home_team, incumbent_name, keywords):
    print("Generating search queries using the LLM...")
    query_prompt = f"Generate search queries for the following prompt. Ensure to include the incumbent ({incumbent_name}) and home team ({home_team}) where relevant:\n\n{prompt}"
    queries_response = get_ollama_response(query_prompt, model=MODEL)
    queries = queries_response.split('\n')
    filtered_queries = [
        query.strip() + " " + " ".join(keywords)
        for query in queries if query.strip() and any(keyword in query for keyword in [incumbent_name, home_team] + keywords)
    ]
    return filtered_queries

def web_search(query):
    print(f"Performing web search for query: {query}")
    search_query = query.replace(" ", "+")
    search_url_full = f"{SEARCH_URL}?q={search_query}&format=json"
    print(f"Web search URL: {search_url_full}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(search_url_full, headers=headers)
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

def extract_pdf_content(pdf_content):
    try:
        pdf_text = ""
        pdf_file = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        if pdf_file.is_encrypted:
            pdf_file.decrypt('')
        for page_num in range(len(pdf_file.pages)):
            try:
                page_text = pdf_file.pages[page_num].extract_text()
                if page_text:
                    pdf_text += page_text
                else:
                    print(f"Warning: Page {page_num} is empty or could not be read.")
            except Exception as e:
                print(f"Warning: Page {page_num} could not be read. Error: {e}")
                continue
        return pdf_text
    except Exception as e:
        print(f"Error extracting PDF content: {e}")
    return ""

print("Loading SOW text...")
with open(f'{DATA_FOLDER}/{SOW_FILE}', 'r', encoding='utf-8') as sow_file:
    sow_text = sow_file.read()
print("SOW text loaded.")

print("Loading company's capabilities...")
with open(f'{DATA_FOLDER}/{CAPABILITIES_FILE}', 'r', encoding='utf-8') as capabilities_file:
    capabilities_text = capabilities_file.read()
print("Company's capabilities loaded.")

print("Loading questions from file...")
with open(f'{DATA_FOLDER}/{QUESTIONS_FILE}', 'r', encoding='utf-8') as questions_file:
    questions = [line.strip() for line in questions_file.readlines()]
print("Questions loaded.")

add_document_to_vector_store(client, "documents", "sow", sow_text)
add_document_to_vector_store(client, "documents", "capabilities", capabilities_text)

def build_context_from_vector_store(client, collection_name):
    context = ""
    scroll = {"offset": 0}
    while True:
        res = client.scroll(collection_name=collection_name, offset=scroll["offset"], limit=100)
        if not res["points"]:
            break
        for point in res["points"]:
            context += point["payload"]["text"] + "\n"
        scroll["offset"] += len(res["points"])
    return context

os.makedirs(RESULTS_FOLDER, exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
result_folder = f"{RESULTS_FOLDER}/{timestamp}"
os.makedirs(result_folder, exist_ok=True)
result_file = f"{result_folder}/QandA_results_{timestamp}.md"
web_data_file = f"{result_folder}/Web_data_{timestamp}.md"
prompt_file = f"{result_folder}/Prompts_{timestamp}.md"

with open(result_file, 'w', encoding='utf-8') as f:
    f.write(f"# Q&A Results ({timestamp})\n")
with open(web_data_file, 'w', encoding='utf-8') as f:
    f.write(f"# Web Data ({timestamp})\n")
with open(prompt_file, 'w', encoding='utf-8') as f:
    f.write(f"# Prompts ({timestamp})\n")

def process_web_content(web_links, num_links=NUM_LINKS):
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

    cleaned_web_content = summarize_and_extract_key_points(web_content)
    return web_content_text, cleaned_web_content, used_links

def summarize_and_extract_key_points(web_contents):
    print("Summarizing and extracting key points from web contents...")
    summary = ""
    for content in web_contents:
        if len(content.strip()) > 0:
            prompt = f"Summarize the following content:\n\n{content}"
            summarized_text = get_ollama_response(prompt)
            summary += summarized_text + "\n"
    return summary

def get_ollama_response(prompt, model=MODEL):
    headers = {'Content-Type': 'application/json'}
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(SERVER_URL, headers=headers, json=payload)
        response.raise_for_status()
        response_data = response.json()
        return response_data.get('response', 'No response found')
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Ollama server: {e}")
        return "Error communicating with Ollama server"

initial_queries_prompt = f"Generate search queries to understand the incumbent's capabilities, comparative analysis, and specific strengths and weaknesses. Ensure to include the incumbent ({INCUMBENT_NAME}) and home team ({HOME_TEAM})."
initial_queries = generate_search_queries(initial_queries_prompt, HOME_TEAM, INCUMBENT_NAME, WEB_SEARCH_KEYWORDS)

initial_info = []
initial_links = []

for query in initial_queries:
    info, links = web_search(query)
    initial_info.extend(info)
    initial_links.extend(links)

initial_web_content_text, initial_cleaned_web_content, initial_used_links = process_web_content(initial_links)

context = f"SOW: {sow_text}\n\nCapabilities: {capabilities_text}\n\nInitial Web Summary: {initial_cleaned_web_content}\n\n"

add_document_to_vector_store(client, "documents", "initial_context", context)

# Add cleaned web content to the vector store
def add_cleaned_web_content_to_vector_store(client, cleaned_content, used_links):
    for idx, content in enumerate(cleaned_content.split("\n\n")):
        if content.strip():
            inputs = tokenizer(content, return_tensors="pt", truncation=True, max_length=512)
            outputs = embedding_model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()

            point = {
                "id": str(uuid.uuid4()),  # Use UUID for point ID
                "vector": embedding.tolist(),
                "payload": {"text": content, "source_url": used_links[idx] if idx < len(used_links) else None}
            }

            try:
                client.upsert(
                    collection_name="documents",
                    points=[point]
                )
            except Exception as e:
                print(f"Error upserting point {point['id']}: {e}")

add_cleaned_web_content_to_vector_store(client, initial_cleaned_web_content, initial_used_links)

answers = {}
for i, question in enumerate(questions):
    start_time = time.time()
    print(f"\nProcessing question {i+1}/{len(questions)}: {question}")

    if "}" in question:
        tag, actual_question = question.split("}", 1)
        tag = tag.strip("{").strip()
        actual_question = actual_question.strip()
    else:
        tag = "general"
        actual_question = question.strip()

    if tag == "incumbent":
        question_queries_prompt = f"Generate search queries for the following question relative to the incumbent ({INCUMBENT_NAME}). Ensure to include the incumbent ({INCUMBENT_NAME}) and home team ({HOME_TEAM}): {actual_question}"
    elif tag == "process":
        question_queries_prompt = f"Generate search queries for the following question related to process. Ensure to include the incumbent ({INCUMBENT_NAME}) and home team ({HOME_TEAM}): {actual_question}"
    else:
        question_queries_prompt = f"Generate search queries for the following question. Ensure to include the incumbent ({INCUMBENT_NAME}) and home team ({HOME_TEAM}): {actual_question}"

    question_queries = generate_search_queries(question_queries_prompt, HOME_TEAM, INCUMBENT_NAME, WEB_SEARCH_KEYWORDS)

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

    web_content_text, cleaned_web_content, used_links = process_web_content(web_links)

    # Add cleaned web content for the question to the vector store
    add_cleaned_web_content_to_vector_store(client, cleaned_web_content, used_links)

    print("Waiting to generate answer...")
    prompt_context = f"SOW: {sow_text}\n\nCapabilities: {capabilities_text}\n\nInitial Web Summary: {initial_cleaned_web_content}\n\nWeb Summary: {cleaned_web_content}\n\n"
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
    context += f"\nQ: {actual_question}\nA: {answer}\nWeb Info: {web_info_text}\nWeb Links: {web_links_text}\nWeb Content: {web_content_text}\nCleaned Web Content: {cleaned_web_content}\nWeb Hits: {num_web_hits}\n"
    print(f"Answer generated and added to context for question {i+1}\n" + "-" * 50)

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

    end_time = time.time()
    processing_time = end_time - start_time

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

with open(result_file, 'r+', encoding='utf-8') as f:
    content = f.read()
    f.seek(0, 0)
    f.write(f"# Summary Statistics\n")
    f.write(f"- Total Questions: {len(questions)}\n")
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
