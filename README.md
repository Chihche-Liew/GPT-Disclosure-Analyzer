# GPT-Disclosure-Analyzer
A high-level Python class for extracting and analyzing selected sections from SEC EDGAR filings (e.g., 10-K forms) using keyword filtering and OpenAI's batch GPT APIs.  This tool is designed for scalable, automated document analysis on corporate disclosures using a hybrid of traditional filtering and large language models (LLMs).

`GPTDisclosureAnalyzer` enables the end-to-end pipeline for:

- Querying and downloading filings from the SEC EDGAR database via SEC-API.io

- Cleaning and extracting specific "items" (e.g., Item 1, Item 7, etc.) from filings

- Identifying relevant paragraphs using custom keyword lists (unigram/bigram supported)

- Submitting paragraph batches to the OpenAI Chat API (via /v1/chat/completions)

- Saving the raw GPT responses for later analysis (JSON-parsed)

- The class is modular and supports fine-grained configuration of filing type, date range, item extraction, batching, and model cost estimation.

## Features
- SEC Filing Querying via sec-api (supports 10-K, 10-Q and 8-K)

- Paragraph Extraction using user-defined keywords

- Batch LLM Annotation using OpenAI GPT models (e.g., gpt-4o-mini)

- Output Management with raw JSONL response saving

- Cost Estimation for prompt + completion token usage

- Fully Reproducible, file-based pipeline (input_batch/ and output_batch/)

## Tested Environment
- OS: Linux (Ubuntu 22.04)
- Python:	3.12
- CPU: Intel Xeon Gold 6248R @ 3.00GHz
- RAM: 753 GB
- SEC-API Access: Required
- OpenAI API Access: Required

## Dependencies
Install with:
```
pip install pandas numpy openai tqdm sec-api
```

## Usage
```{python}
from gpt_disclosure_analyzer import GPTDisclosureAnalyzer

analyzer = GPTDisclosureAnalyzer(
    sec_api_key="your_sec_api_key",
    openai_api_key="your_openai_key",
    keyword_csv_path="keywords.csv",       # Must contain a 'keyword' column
    prompt_file_path="prompt.txt",         # Prompt prepended to each paragraph
    start_year=2021,
    end_year=2022,
    form_type="10-K",
    items=['1'],                           # SEC items (e.g., Item 1, 1A, 7, etc.)
)

result_df = analyzer.process_filings(gpt=True)
```
This will:

- Download filings

- Extract and clean specified items

- Filter relevant paragraphs

- Submit to GPT in batches

- Save raw JSON responses in item_1_anno, etc.

## Example Output (DataFrame Columns)
| Column Name       | Description                        |
| ----------------- | ---------------------------------- |
| `item_1_text`     | Cleaned full text of item          |
| `item_1_count`    | Word count of item                 |
| `item_1_keywords` | Matched keywords in this item      |
| `item_1_paras`    | List of matched paragraphs         |
| `item_1_anno`     | Raw GPT annotations (one per para) |

## Notes
- Does not parse or interpret GPT output — raw JSON is preserved for downstream use

- To change target analysis focus, just update the `keywords.csv` and `prompt.txt`

- Compatible with batch-completion endpoints only (not streaming/completion-only)

## Citations
https://sec-api.io/docs/query-api/python-example
