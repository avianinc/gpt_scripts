Certainly! Here is a `README.md` that thoroughly explains how the code works:

```markdown
# Q&A Generation and Analysis Tool

This tool automates the process of generating answers to a set of questions related to a solicitation statement of work (SOW), competitive analysis, and proposal strategy. It performs web searches to gather relevant information, summarizes the content, and uses a language model to provide answers. Additionally, it generates a final summary with insights about the opportunity, incumbent strengths and weaknesses, Home Team strengths and weaknesses, and recommendations.

## Table of Contents

- [Requirements](#requirements)
- [Configuration](#configuration)
- [Running the Tool](#running-the-tool)
- [File Structure](#file-structure)
- [Code Overview](#code-overview)
- [Output](#output)

## Requirements

- Python 3.7+
- Required Python packages:
  - `requests`
  - `beautifulsoup4`
  - `datetime`
  - `transformers`
  - `os`

## Configuration

The tool requires a `config.json` file to be present in the same directory as the script. The configuration file should contain the following fields:

```json
[
  {
    "model": "llama3",
    "download_model": "llama3",
    "data_folder": "data",
    "capabilities_file": "capabilities.txt",
    "sow_file": "sow.txt",
    "questions_file": "questions.txt",
    "num_links": 5,
    "server_url": "http://localhost:11434/api/generate",
    "search_url": "http://localhost:8080/search",
    "web_search_company_name": "The incumbent is Draken International",
    "input_prompt": "You are a helpful server capable of helping me understand how to better create responses to statements of work (SOW). You provide thorough analysis of capabilities, statements of work, and questions to assist in proposal writing. Your responses should include insights from the context provided and relevant external information gathered from web searches.",
    "home_team": "Home Team",
    "incumbent_name": "Draken International"
  }
]
```

### Explanation of Configuration Fields

- `model`: The name of the language model to use.
- `download_model`: The model to be downloaded.
- `data_folder`: Folder containing data files.
- `capabilities_file`: File containing the Home Team's capabilities.
- `sow_file`: File containing the solicitation statement of work (SOW).
- `questions_file`: File containing the list of questions.
- `num_links`: Number of web links to process per search query.
- `server_url`: URL of the server running the language model.
- `search_url`: URL of the web search service.
- `web_search_company_name`: The incumbent company name for web search queries.
- `input_prompt`: The prompt provided to the language model for generating answers.
- `home_team`: Name of the Home Team.
- `incumbent_name`: Name of the incumbent company.

## Running the Tool

1. Ensure you have all the required Python packages installed.
2. Prepare the `config.json` file and place it in the same directory as the script.
3. Place the `capabilities.txt`, `sow.txt`, and `questions.txt` files in the specified `data_folder`.
4. Run the script using the following command:
   ```sh
   python generate_qa.py
   ```

## File Structure

- `config.json`: Configuration file containing settings and parameters.
- `data/`: Directory containing the following files:
  - `capabilities.txt`: Home Team's capabilities.
  - `sow.txt`: Solicitation statement of work.
  - `questions.txt`: List of questions.
- `generate_qa.py`: Main script to run the tool.
- `results/`: Directory where the results will be saved.

## Code Overview

### Importing Modules

The script starts by importing necessary modules for web requests, JSON handling, HTML parsing, and other utilities.

### Loading Configuration

The configuration is loaded from the `config.json` file, and necessary parameters are extracted.

### Initial Web Searches

The tool performs initial web searches to gather general information about the incumbent's capabilities, comparisons, and strengths and weaknesses. This information is processed and summarized.

### Processing Questions

For each question in the `questions.txt` file:
1. The script determines the type of question based on tags (`{incumbent}`, `{process}`, etc.).
2. It performs a web search specific to the question and the incumbent or process.
3. The web content is extracted, summarized, and used to generate an answer using the language model.
4. Results, including web content and links, are saved for each question.

### Generating Final Summary

After answering all questions, the tool generates a final summary with insights about the opportunity, incumbent strengths and weaknesses, Home Team strengths and weaknesses, and recommendations.

### Saving Results

Results are saved in the `results/` directory with a timestamp, including:
- Q&A results
- Web data
- Prompts used for generating answers

## Output

The output includes the following files saved in the `results/` directory with a timestamp:
- `QandA_results_{timestamp}.md`: Contains answers to the questions, statistical summary, and final summary.
- `Web_data_{timestamp}.md`: Contains the web data gathered for each question.
- `Prompts_{timestamp}.md`: Contains the prompts sent to the language model for each question.

### Example Output Structure

```plaintext
results/
├── 20240619_123456/
│   ├── QandA_results_20240619_123456.md
│   ├── Web_data_20240619_123456.md
│   ├── Prompts_20240619_123456.md
```

### Example `QandA_results_{timestamp}.md`

```markdown
# Q&A Results (20240619_123456)

# Summary Statistics
- Total Questions: 20
- Total Web Hits: 100
- Total Processing Time: 500.00 seconds

# Configuration Variables
- Model: llama3
- Data Folder: data
- Capabilities File: capabilities.txt
- SOW File: sow.txt
- Questions File: questions.txt
- Results Folder: results
- Number of Links: 5
- Server URL: http://localhost:11434/api/generate
- Search URL: http://localhost:8080/search
- Web Search Company Name: The incumbent is Draken International
- Input Prompt: You are a helpful server capable of helping me understand how to better create responses to statements of work (SOW). You provide thorough analysis of capabilities, statements of work, and questions to assist in proposal writing. Your responses should include insights from the context provided and relevant external information gathered from web searches.
- Home Team: Home Team
- Incumbent Name: Draken International

# BLUF (Bottom Line Up Front) Summary
[Final summary content]

## Q: {actual_question_1}
**A:** {answer_1}
**References:**
- {link_1}
- {link_2}
...

## Q: {actual_question_2}
**A:** {answer_2}
**References:**
- {link_1}
- {link_2}
...
```

## Contact

For any questions or issues, please contact John DeHart at jdehart@avian.com.
```

This `README.md` provides a comprehensive guide on how to use the tool, including setup, configuration, running the tool, understanding the code, and interpreting the output.

Process Flow

| Step | Description |
|------|-------------|
| 1    | **Load Configuration**: Load the configuration from the `config.json` file. |
| 2    | **Create Results Folder**: Create a results folder if it does not exist. |
| 3    | **Define `get_ollama_response` Function**: Define a function to get a response from the Ollama server. |
| 4    | **Define `generate_search_queries` Function**: Define a function to generate search queries using the language model. |
| 5    | **Define `summarize_and_extract_key_points` Function**: Define a function to summarize and extract key points from web contents. |
| 6    | **Define `web_search` Function**: Define a function to perform a web search and get top results. |
| 7    | **Define `extract_content` Function**: Define a function to extract content from a URL with error handling and retries. |
| 8    | **Define `extract_pdf_urls` Function**: Define a function to extract PDF URLs from a webpage. |
| 9    | **Load SOW Text**: Load the Statement of Work (SOW) text from a file. |
| 10   | **Load Company’s Capabilities**: Load the company's capabilities from a file. |
| 11   | **Load Questions**: Load the questions from a file. |
| 12   | **Combine SOW and Capabilities into Context**: Combine the SOW and capabilities into a single context. |
| 13   | **Timestamp for Filenames and Folder**: Create a timestamp for filenames and result folder. |
| 14   | **Initialize Files**: Initialize result files for Q&A results, web data, and prompts. |
| 15   | **Define `process_web_content` Function**: Define a function to process web content and summarize it. |
| 16   | **Initialize Summary Statistics**: Initialize summary statistics for total questions, web hits, and processing time. |
| 17   | **Generate Initial Queries**: Generate initial search queries about the incumbent. |
| 18   | **Perform Initial Web Search**: Perform a web search using the initial queries and process the content. |
| 19   | **Iterate Over Questions**: Iterate over each question to get answers and update the context. |
| 20   | **Generate Final Summary**: Generate a final summary with context including incumbent and home team names. |
| 21   | **Save Summary Statistics and Configuration**: Save the summary statistics and configuration variables to the results file. |
