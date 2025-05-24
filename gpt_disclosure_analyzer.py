import re
import os
import json
import time
import html
import warnings
import pandas as pd
import numpy as np
from openai import OpenAI
from tqdm.notebook import tqdm
from sec_api import QueryApi, ExtractorApi
warnings.filterwarnings("ignore")


class GPTDisclosureAnalyzer:
    def __init__(
            self,
            sec_api_key: str,
            openai_api_key: str,
            keyword_csv_path: str,
            prompt_file_path: str,
            start_year: int,
            end_year: int,
            form_type: str = "10-K",
            items=None,
            limit: int = 200,
            batch_size_limit: int = 100 * 1024 * 1024,
            input_base_path: str = "input_batch",
            output_base_path: str = "output_batch",
            input_price_per_mtok: float = 0.15,
            output_price_per_mtok: float = 0.60
    ):
        if items is None:
            items = ['1']
        self.sec_api_key = sec_api_key
        self.openai_api_key = openai_api_key
        self.keyword_csv_path = keyword_csv_path
        self.prompt_file_path = prompt_file_path
        self.start_year = start_year
        self.end_year = end_year
        self.form_type = form_type
        self.limit = limit

        self.keywords = self._load_keywords()
        self.keywords_lc = {kw.lower() for kw in self.keywords}
        self.gpt_prompt = self._load_gpt_prompt() if prompt_file_path else ""

        if openai_api_key:
            self.client = OpenAI(api_key=openai_api_key)
        else:
            self.client = None

        self.items = items
        self.query_api = QueryApi(sec_api_key)
        self.extractor_api = ExtractorApi(sec_api_key)

        self.batch_size_limit = batch_size_limit
        self.input_base_path = input_base_path
        self.output_base_path = output_base_path
        self.INPUT_PRICE = input_price_per_mtok
        self.OUTPUT_PRICE = output_price_per_mtok

    def _load_keywords(self) -> list:
        df = pd.read_csv(self.keyword_csv_path)
        return list(df["keyword"])

    def _load_gpt_prompt(self) -> str:
        with open(self.prompt_file_path, "r", encoding="utf-8") as f:
            prompt = f.read()
        return prompt

    @staticmethod
    def count_words(text: str) -> int:
        return len(text.split())

    @staticmethod
    def clean_text(text: str) -> str:
        text = html.unescape(text)
        text = text.replace("\n", " ")
        text = re.sub(r'[\xa0\u200b\u202f\u3000\ufeff]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def extract_related_paras(self, text):
        if not text:
            return None

        words = text.split()

        if not words:
            return None

        kws = {kw.lower() for kw in self.keywords}
        paras = []

        for i in range(len(words)):
            unigram = words[i].lower()
            if unigram in kws:
                start = max(0, i - self.limit)
                end = min(len(words), i + 1 + self.limit)
                paras.append({"keyword": unigram, "text": " ".join(words[start:end])})

            if i < len(words) - 1:
                bigram = f"{words[i].lower()} {words[i+1].lower()}"
                if bigram in kws:
                    start = max(0, i - self.limit)
                    end = min(len(words), i + 2 + self.limit)
                    paras.append({"keyword": bigram, "text": " ".join(words[start:end])})

        return paras if paras else None

    def fetch_filings(self) -> pd.DataFrame:
        start_year = self.start_year
        end_year = self.end_year
        form_type = self.form_type
        filing_index = []
        print("Fetching filings' index from EDGAR...")
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                query = (
                    f'formType:"{form_type}" AND NOT formType:"{form_type}/A" AND NOT formType:"{form_type}A" AND '
                    f'periodOfReport:[{year}-{month:02d}-01 TO {year}-{month:02d}-31]'
                )
                for from_batch in range(0, 9800, 200):
                    query_payload = {
                        "query": {
                            "query_string": {"query": query, "time_zone": "America/New_York"}
                        },
                        "from": from_batch,
                        "size": 200,
                        "sort": [{"filedAt": {"order": "desc"}}]
                    }
                    response = self.query_api.get_filings(query_payload)

                    if len(response.get("filings", [])) == 0:
                        break

                    for filing in response["filings"]:
                        cik = filing.get("cik", "")
                        filed_at = filing.get("filedAt", "").split("T")[0]
                        filing_url = filing.get("linkToFilingDetails", "")
                        company_name = filing.get("companyName", "")
                        period_of_report = filing.get("periodOfReport", "")
                        try:
                            sic = filing["entities"][0].get("sic", "")
                            sic = re.findall(r'\d+', sic)[0]
                        except IndexError:
                            sic = None
                        except KeyError:
                            sic = None
                        filing_index.append({
                            "cik": cik,
                            "file_date": filed_at,
                            "report_date": period_of_report,
                            "company_name": company_name,
                            "sic": sic,
                            "filing_url": filing_url
                        })

        return pd.DataFrame(filing_index)

    def parse_filings(self, filing_url: str) -> dict:
        items = self.items.copy()
        items_text = {}
        for item in items:
            try:
                items_text[f'item_{item.lower()}_text'] = self.extractor_api.get_section(filing_url, item, "text")
                items_text[f'item_{item.lower()}_text'] = self.clean_text(items_text[f'item_{item.lower()}_text'])
                items_text[f'item_{item.lower()}_count'] = self.count_words(items_text[f'item_{item.lower()}_text'])
            except Exception as e:
                print(f"Error parsing item {item} of {filing_url}: {e}")
        return items_text

    def _prepare_batch_input(self, paragraphs_to_analyze: list) -> list:
        batch_input_messages = []
        for para_info in paragraphs_to_analyze:
            custom_id = f"filing-{para_info['filing_idx']}_item-{para_info['item']}_para-{para_info['para_list_idx']}"
            message = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": 'gpt-4o-mini',
                    "messages": [
                        # Optional:
                        # {
                        #     'role': 'system',
                        #     'content': ""
                        # },
                        {
                            "role": "user",
                            "content": self.gpt_prompt + "\n\n" + para_info['text'],
                        }
                    ],
                    "temperature": 0,
                    "response_format": { "type": "json_object" },
                    "max_tokens": 500
                }
            }
            batch_input_messages.append(message)
        return batch_input_messages

    @staticmethod
    def _save_input_file(input_file_path: str, batch_input_file: list):
        os.makedirs(os.path.dirname(input_file_path), exist_ok=True)
        with open(input_file_path, 'w', encoding='utf-8') as f:
            for m in batch_input_file:
                f.write(json.dumps(m, ensure_ascii=False) + '\n')
        print(f"Saved {input_file_path} with {len(batch_input_file)} messages.")

    def submit_batch(self, input_file_path: str, file_index: int):
        if not self.client:
            raise ValueError("OpenAI client not initialized. Check openai_api_key.")

        print(f"Submitting input file {input_file_path} for batch processing...")
        batch_input_file_obj = self.client.files.create(
            file=open(input_file_path, 'rb'),
            purpose='batch'
        )
        batch_input_file_id = batch_input_file_obj.id

        batch = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": f"_{file_index}"
            }
        )
        print(f"Submitted batch {batch.id} using file {batch_input_file_id}.")
        return batch

    def _save_output_file(self, batch, file_index: int):
        print(f"Downloading output file {batch.output_file_id} for batch {batch.id}...")
        try:
            output_file_content = self.client.files.content(batch.output_file_id)
            output_file_path = os.path.join(self.output_base_path, f"output_batch_{file_index}.jsonl")
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            with open(output_file_path, 'wb') as f:
                f.write(output_file_content.content)
            print(f"Saved output file: {output_file_path}")
            return output_file_path
        except Exception as e:
            print(f"Error downloading or saving output file for batch {batch.id}: {e}")
            return None

    def _monitor_batches(self, batch_objs_with_index: list) -> list:
        batch_info_map = {batch.id: {'batch_obj': batch, 'file_index': file_index}
                          for batch, file_index in batch_objs_with_index}
        batch_ids_to_monitor = list(batch_info_map.keys())
        completed_output_files = []

        print(f"Monitoring {len(batch_ids_to_monitor)} batches...")

        while batch_ids_to_monitor:
            time.sleep(120)
            print(f"Polling batch statuses. Remaining: {len(batch_ids_to_monitor)}...")
            remaining_ids = []

            for batch_id in batch_ids_to_monitor:
                try:
                    updated_batch = self.client.batches.retrieve(batch_id)
                    batch_info_map[batch_id]['batch_obj'] = updated_batch
                    status = updated_batch.status
                    print(f"Batch {batch_id} status: {status}")

                    if status == 'completed':
                        file_index = batch_info_map[batch_id]['file_index']
                        output_path = self._save_output_file(updated_batch, file_index)
                        if output_path:
                            completed_output_files.append(output_path)

                    elif status in ['failed', 'cancelled', 'expired']:
                        print(f"Batch {batch_id} ended with status: {status}")

                    else:
                        remaining_ids.append(batch_id)

                except Exception as e:
                    print(f"Error monitoring batch {batch_id}: {e}")
                    remaining_ids.append(batch_id) # Keep trying

            batch_ids_to_monitor = remaining_ids

        print("Finished monitoring all batches.")
        return completed_output_files

    def _process_batch_results(self, output_files: list) -> tuple[pd.DataFrame, float]:
        results_list = []
        total_cost = 0
        print("Processing batch results...")

        for output_file_path in output_files:
            try:
                with open(output_file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            result_json = json.loads(line.strip())
                            if result_json.get('response') and result_json['response'].get('body'):
                                body = result_json['response']['body']
                                custom_id = result_json.get('custom_id', None)
                                usage = body.get('usage', {})
                                completion_tokens = usage.get('completion_tokens', 0)
                                prompt_tokens = usage.get('prompt_tokens', 0)

                                request_cost = (prompt_tokens / 1_000_000 * self.INPUT_PRICE) + \
                                               (completion_tokens / 1_000_000 * self.OUTPUT_PRICE)
                                total_cost += request_cost

                                try:
                                    reply_content = body['choices'][0]['message']['content']
                                    reply_content = reply_content.replace("```json", "").replace("```", "").strip()
                                    parsed_response = json.loads(reply_content)

                                    results_list.append({
                                        'custom_id': custom_id,
                                        'parsed_response': parsed_response,
                                        'request_cost': request_cost,
                                        'prompt_tokens': prompt_tokens,
                                        'completion_tokens': completion_tokens
                                    })
                                except (json.JSONDecodeError, IndexError, KeyError) as e:
                                    print(f"Warning: Failed to parse response content for custom_id {custom_id} in {output_file_path}: {e}")
                                    results_list.append({
                                        'custom_id': custom_id,
                                        'parsed_response': {'error': f'Failed to parse JSON: {e}'},
                                        'request_cost': request_cost,
                                        'prompt_tokens': prompt_tokens,
                                        'completion_tokens': completion_tokens
                                    })
                            else:
                                print(f"Warning: Unexpected structure in result line from {output_file_path}: {result_json}")
                        except json.JSONDecodeError as e:
                            print(f"Warning: Failed to parse line in {output_file_path}: {line.strip()}. Error: {e}")
            except FileNotFoundError:
                print(f"Error: Output file not found at {output_file_path}")
            except Exception as e:
                print(f"An unexpected error occurred while processing {output_file_path}: {e}")


        results_df = pd.DataFrame(results_list)
        print(f"Finished processing results from {len(output_files)} files.")
        return results_df, total_cost

    @staticmethod
    def _parse_custom_id(custom_id: str) -> tuple[int, str, int] | tuple[None, None, None]:
        match = re.match(r'filing-(\d+)_item-(.+)_para-(\d+)', custom_id)
        if match:
            try:
                filing_idx = int(match.group(1))
                item = match.group(2)
                para_list_idx = int(match.group(3))
                return filing_idx, item, para_list_idx
            except ValueError as e:
                print(f"Error parsing parts of custom_id {custom_id}: {e}")
                return None, None, None
        else:
            print(f"Warning: Custom ID format not recognized: {custom_id}")
            return None, None, None


    def _run_batch_analysis(self, batch_input_messages: list) -> tuple[pd.DataFrame, float]:

        current_batch_size = 0
        file_index = 0
        current_batch_input_messages = []
        batch_objs_to_monitor = []

        print(f"Preparing {len(batch_input_messages)} messages for batch submission...")

        for msg in tqdm(batch_input_messages, desc="Preparing and submitting batches"):
            message_str = json.dumps(msg, ensure_ascii=False)
            message_size = len(message_str.encode('utf-8')) + 1

            if current_batch_size + message_size > self.batch_size_limit and current_batch_input_messages:
                input_file_path = os.path.join(self.input_base_path, f"input_batch_{file_index}.jsonl")
                self._save_input_file(input_file_path, current_batch_input_messages)

                batch = self.submit_batch(input_file_path, file_index)
                batch_objs_to_monitor.append((batch, file_index))

                current_batch_input_messages = []
                current_batch_size = 0
                file_index += 1

            current_batch_input_messages.append(msg)
            current_batch_size += message_size

        if current_batch_input_messages:
            input_file_path = os.path.join(self.input_base_path, f"input_batch_{file_index}.jsonl")
            self._save_input_file(input_file_path, current_batch_input_messages)

            batch = self.submit_batch(input_file_path, file_index)
            batch_objs_to_monitor.append((batch, file_index))

        print(f"All input files saved and batches submitted. Total batches: {len(batch_objs_to_monitor)}")

        output_files = self._monitor_batches(batch_objs_to_monitor)

        results_df, total_cost = self._process_batch_results(output_files)

        print(f"Estimated total cost for batch analysis: ${total_cost:.4f}")

        return results_df, total_cost

    def process_filings(self, gpt=False):
        filings_df = self.fetch_filings()
        print(f"Fetched index for {len(filings_df)} filings.")

        if filings_df.empty:
            print("No filings found. Exiting.")
            return pd.DataFrame()

        for item in self.items:
            filings_df[f'item_{item.lower()}_text'] = ""
            filings_df[f'item_{item.lower()}_count'] = 0
            filings_df[f'item_{item.lower()}_keywords'] = None
            filings_df[f'item_{item.lower()}_paras'] = None
            filings_df[f'item_{item.lower()}_anno'] = None

        paragraphs_to_analyze = []
        processed_filings_df = filings_df.copy()

        print("Parsing filings and extracting paragraphs...")
        for idx, row in tqdm(processed_filings_df.iterrows(), total=len(processed_filings_df), desc="Parsing and Extracting"):
            filing_url = row['filing_url']
            parsed_items = self.parse_filings(filing_url)

            for item in self.items:
                item_text = parsed_items.get(f'item_{item.lower()}_text', None)
                item_count = parsed_items.get(f'item_{item.lower()}_count', 0)

                processed_filings_df.at[idx, f'item_{item.lower()}_text'] = item_text
                processed_filings_df.at[idx, f'item_{item.lower()}_count'] = item_count

                paras = self.extract_related_paras(item_text)
                if paras:
                    processed_filings_df.at[idx, f'item_{item.lower()}_keywords'] = list({para["keyword"] for para in paras})
                    para_texts = [para["text"] for para in paras]
                    processed_filings_df.at[idx, f'item_{item.lower()}_paras'] = para_texts

                    if gpt and self.client:
                        for para_list_idx, para_text in enumerate(para_texts):
                            paragraphs_to_analyze.append({
                                'text': para_text,
                                'filing_idx': idx,
                                'item': item.lower(),
                                'para_list_idx': para_list_idx
                            })

        print(f"Extracted a total of {len(paragraphs_to_analyze)} paragraphs for potential analysis.")

        if gpt and self.client and paragraphs_to_analyze:
            print("Initiating GPT batch analysis...")
            batch_input_messages = self._prepare_batch_input(paragraphs_to_analyze)
            results_df, total_cost = self._run_batch_analysis(batch_input_messages)

            print("Saving raw GPT responses to DataFrame...")

            if not results_df.empty:
                anno_map = {row['custom_id']: row['parsed_response'] for _, row in results_df.iterrows()}

                for idx, row in tqdm(processed_filings_df.iterrows(), total=len(processed_filings_df), desc="Attaching raw annotations"):
                    for item in self.items:
                        para_list = row.get(f'item_{item.lower()}_paras', None)
                        if para_list is not None:
                            annotations = []
                            for para_idx in range(len(para_list)):
                                custom_id = f"filing-{idx}_item-{item.lower()}_para-{para_idx}"
                                annotations.append(anno_map.get(custom_id))
                            processed_filings_df.at[idx, f'item_{item.lower()}_anno'] = annotations

        elif gpt and not self.client:
            print("Skipping GPT analysis because OpenAI client is not initialized.")
        elif gpt and not paragraphs_to_analyze:
            print("Skipping GPT analysis because no relevant paragraphs were extracted.")
        else:
            print("GPT analysis not requested.")

        return processed_filings_df