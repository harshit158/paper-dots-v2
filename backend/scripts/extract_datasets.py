import os
import json
import instructor
from pydantic import BaseModel, Field
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import fitz
import requests
import re
from io import BytesIO
import cohere
import google.generativeai as genai
from tqdm import tqdm
from multiprocessing import Pool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

from logging_setup import get_logger
logger = get_logger(__name__)

from gliner import GLiNER
MODEL = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

class Entities(BaseModel):
    datasets: list[str] = Field(description = """Names of datasets mentioned 
                                     in the text which are used by ML model / algorithm for 
                                     training.""")

PROMPT = """Extract the names of datasets mentioned in the given text.
Don't give any response if you don't find any datasets

## Text:
# {}
"""
            
def extract_text_from_pdf(url):
    response = requests.get(url)
    pdf_content = BytesIO(response.content)
    document = fitz.Document(stream=pdf_content, filetype="pdf")

    # Extract text from each page
    doc_blocks = []
    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        page_blocks = page.get_text("blocks")
        for block in page_blocks:
            doc_blocks.append(block[4])

    logger.info(f"Total blocks: {len(doc_blocks)}")
    return doc_blocks

def get_page_span(blocks):
    abstract_idx = None
    for i in range(len(blocks)):
        if re.search("^(Abstract|ABSTRACT)", blocks[i]):
            abstract_idx = i
            break
    
    references_idx = None
    for i in range(len(blocks)-1, -1, -1):
        if re.search("^(References|REFERENCES)", blocks[i]):
            references_idx = i
            break
    
    return abstract_idx, references_idx
            
def clean_block(block: str) -> str:
    if re.search("^figure", block, re.IGNORECASE):
        return ""
    
    MIN_BLOCK_LENGTH = 20 #words
    single_line_block = re.sub("-\n", "", block)
    single_line_block = block.replace("\n", " ")
    single_line_block = re.sub(r"-?\d+(?:\.\d+)?", " ", single_line_block)
    if len(single_line_block.split()) < MIN_BLOCK_LENGTH:
        return ""

    block = block.replace("-\n", "")
    block = block.replace("\n", " ")
    return block

def clean_blocks(blocks: list[str]):
    abstract_idx, references_idx = get_page_span(blocks)

    assert abstract_idx is not None and references_idx is not None
    
    blocks = blocks[abstract_idx+1: references_idx] 
    cleaned_blocks = []
    for block in blocks:
        block = clean_block(block)
        if block:
            cleaned_blocks.append(block)
    
    logger.info(f"Total cleaned blocks: {len(cleaned_blocks)}")
    return cleaned_blocks

def get_response(client, text: str):
    return client.messages.create(
        messages=[
            {"role": "system",
            "content": """You're a powerful language model that has been specialized for Named Entity Recognition."""},
            {"role": "user", "content": PROMPT.format(text)}],
    
        response_model=Entities,
    )
    
def get_response_cohere(client, text: str):
    return client.chat.completions.create(
        model="command-r-08-2024",
        messages=[
            {"role": "system",
            "content": """You're a powerful language model that has been finetuned for Named Entity Recognition."""},
            {"role": "user", "content": PROMPT.format(text)}],
        
        response_model=Entities,
)

def get_gemini_client():
    client = instructor.from_gemini(
            client=genai.GenerativeModel(model_name="models/gemini-1.5-flash-latest"),
            mode=instructor.Mode.GEMINI_JSON,
    )
    return client

def get_cohere_client():
    client = instructor.from_cohere(
            cohere.Client()
    )
    
    return client

# def extract(cleaned_blocks: list[str]):
#     client = get_gemini_client()
#     # client = get_cohere_client()
    
#     with ThreadPoolExecutor(max_workers=10) as executor:
#         responses = list(executor.map(lambda block: get_response(client, block), cleaned_blocks))
    
#     # responses = []
#     # for i, block in enumerate(cleaned_blocks):
#     #     logger.info(f"Processing block {i}")
#     #     responses.append(get_response(client, block))
    
#     datasets = sum([x.datasets for x in responses], [])
#     # methods = sum([x.methods for x in responses], [])
    
#     return list(set(datasets))

def extract(text: str):
    labels = ["dataset_name", "model_name", "benchmark", "others"]
    datasets = []
    entities = MODEL.predict_entities(text, labels)
    for entity in entities:
        if entity["label"] == "dataset_name" and entity["score"] > 0.85 and entity["text"][0].isupper():
            datasets.append(entity["text"])
    
    return list(set(datasets))
       
def split_blocks(blocks: list[str]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=384,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
    )
    all_blocks = []
    for block in blocks:
        splits = text_splitter.create_documents([block])
        texts = [x.page_content for x in splits]
        all_blocks.extend(texts)
        
    return all_blocks
    
def extract_entities(url: str):
    # extract text
    doc_blocks = extract_text_from_pdf(url)
    
    # clean text
    cleaned_blocks = clean_blocks(doc_blocks)
    cleaned_blocks = split_blocks(cleaned_blocks)
    
    # extract entities
    # with Pool(processes=4) as pool:
    #     datasets = pool.map(extract, cleaned_blocks)
    # datasets = list(set(sum(datasets, [])))
    
    datasets = []
    for block in tqdm(cleaned_blocks):
        datasets.extend(extract(block))
    datasets = list(set(datasets))
        
    # datasets = extract(cleaned_blocks)
    logger.info(f"Extracted {len(datasets)} datasets")
    
    return datasets
    
    
def main():
    urls = open(INPUT_PATH).readlines()[:]
    
    try:
        datasets_dict = json.load(open(os.path.join(OUTPUT_DIR, "datasets.json")))
    except:
        datasets_dict = {}
        
    for url in urls:
        logger.info(f"Processing {url}")
        url = url.strip()
        
        datasets = extract_entities(url)
        
        datasets_dict[url] = datasets
        
    json.dump(datasets_dict, open(os.path.join(OUTPUT_DIR, "datasets.json"), "w"))


if __name__ == "__main__":
    INPUT_PATH = "data/input/arxiv_urls.txt"
    OUTPUT_DIR = "data/output"
    main()