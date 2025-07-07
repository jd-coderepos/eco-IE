import os
import json
import torch
import unicodedata
import logging
import re
import spacy
from pathlib import Path
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

# === Configuration ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Paths - update as needed
MODEL_DIR  = "fine_tuned_qwen3_ner"
TXT_PATH   = "data/fine-tuned/txt/(3) Bulmer2024.txt"          # Input .txt file
SCHEMA_TSV = "data/fine-tuned/tsv/(3) Bulmer2024.tsv"  # WebAnno schema .tsv file (headers)
OUTPUT_DIR = "output_ner"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# Prompts (Super strict!)
SYSTEM_PROMPT = "You are an expert in scientific information extraction."
USER_PROMPT_TEMPLATE = """
You are an expert in scientific Named Entity Recognition (NER).

Instructions:
- Given the following text, output ONLY a JSON dictionary mapping each entity phrase (exact text span) to its label.
- DO NOT output any explanations, commentary, <think> tags, or any text before or after the JSON.
- The only valid output is a JSON object. Example:

{{
    "Aotearoa New Zealand": "Locationofstudy",
    "First published: 30 July 2024": "Timeperiodofstudy"
}}

Text: {text}
JSON:
"""

# ─────────────────────────────────────────────────────────────────────────────
def load_model(model_name: str):
    logger.info("torch.cuda.is_available(): %s", torch.cuda.is_available())
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    logger.info("Model loaded on device: %s", next(model.parameters()).device)
    return tokenizer, model

def read_tsv_headers(tsv_path: str) -> List[str]:
    headers: List[str] = []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#T_SP=webanno.custom.'):
                label = line.strip().split('.')[-1].split('|')[0]
                headers.append(label)
    logger.info("WebAnno entity types: %s", headers)
    return headers

def read_text(text_path: str) -> str:
    path = Path(text_path)
    raw = path.read_bytes()
    try:
        text = raw.decode('utf-8')
        logger.info("Decoded text as UTF-8.")
    except UnicodeDecodeError as e:
        logger.warning("UTF-8 decode failed: %s. Trying latin-1.", e)
        text = raw.decode('latin-1', errors='replace')
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r"\s+", " ", text).strip()
    logger.info("Loaded text length: %d chars", len(text))
    return text

# --------- NEW: Robust JSON Extraction ---------
def extract_balanced_json(response: str) -> dict:
    """
    Extract the *largest* well-formed { ... } JSON object from the LLM response.
    Works even with extra explanations or stray braces.
    """
    # Optional: Remove <think> or similar blocks
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

    start = response.find('{')
    if start == -1:
        logger.error("No opening brace found in model output!\nRaw output:\n%s", response)
        return {}
    count = 0
    for i in range(start, len(response)):
        if response[i] == '{':
            count += 1
        elif response[i] == '}':
            count -= 1
            if count == 0:
                json_str = response[start:i+1]
                try:
                    data = json.loads(json_str)
                    logger.info("Parsed JSON with %d items", len(data))
                    return data
                except Exception as e:
                    logger.error("JSON parse error: %s\nBlock: %s", e, json_str)
                    return {}
    logger.error("No balanced closing brace found in output! Start at: %d\n%s", start, response)
    return {}

def annotate_to_json(text: str, tokenizer, model, max_tokens:int = 2048) -> dict:
    user_prompt = USER_PROMPT_TEMPLATE.format(text=text)
    # If using a chat template (Qwen models usually do), keep this structure:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt}
    ]
    # Use chat template if supported, else fallback to prompt only
    if hasattr(tokenizer, "apply_chat_template"):
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        chat_prompt = SYSTEM_PROMPT + "\n" + user_prompt

    inputs = tokenizer([chat_prompt], return_tensors='pt').to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    gen_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)]
    response = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    logger.debug("Raw model response:\n%s", response)
    return extract_balanced_json(response)

def build_spans(span_map: Dict[str, str], text: str) -> List[Tuple[int,int,str]]:
    spans: List[Tuple[int,int,str]] = []
    normalized_text = unicodedata.normalize("NFKC", text)
    normalized_text = re.sub(r"\s+", " ", normalized_text)
    for phrase, label in span_map.items():
        norm_phrase = unicodedata.normalize("NFKC", phrase)
        norm_phrase = re.sub(r"\s+", " ", norm_phrase)
        for m in re.finditer(re.escape(norm_phrase), normalized_text, flags=re.IGNORECASE):
            orig_start = text.find(phrase)
            if orig_start == -1:
                logger.warning("Phrase not found in text: '%s'", phrase)
                continue
            orig_end = orig_start + len(phrase)
            spans.append((orig_start, orig_end, label))
            logger.debug("Matched: '%s' [%d,%d] -> %s", phrase, orig_start, orig_end, label)
    spans.sort(key=lambda x: x[0])
    logger.info("Built %d spans from JSON", len(spans))
    return spans

def json_to_webanno(
    spans: List[Tuple[int,int,str]],
    headers: List[str],
    text: str
) -> List[str]:
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    sentences = list(doc.sents)
    lines: List[str] = ['#FORMAT=WebAnno TSV 3.3']
    for h in headers:
        lines.append(f'#T_SP=webanno.custom.{h}|')
    ann_counter = 1
    active_spans: Dict[str,int] = {}
    prev_label: str = None
    for sent_idx, sent in enumerate(sentences, start=1):
        lines.append(f'\n#Text={sent.text}')
        for tok_idx, tok in enumerate(sent, start=1):
            token_text = tok.text
            start_char = tok.idx
            end_char = start_char + len(token_text)
            label = 'O'
            for (s0, s1, span_label) in spans:
                if start_char >= s0 and end_char <= s1:
                    label = span_label
                    break
            row = [f'{sent_idx}-{tok_idx}', f'{start_char}-{end_char}', token_text]
            for h in headers:
                if label == h:
                    if label != prev_label:
                        active_spans[label] = ann_counter
                        ann_counter += 1
                    row.append(f'*[{active_spans[label]}]')
                else:
                    row.append('_')
            prev_label = label if label != 'O' else None
            lines.append('\t'.join(row))
    logger.info("Conversion complete with %d annotation spans", ann_counter - 1)
    return lines

def save_tsv(lines: List[str], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    logger.info("Saved WebAnno TSV to: %s", out_path)

# === Main ===
if __name__ == '__main__':
    tokenizer, model = load_model(MODEL_DIR)
    text     = read_text(TXT_PATH)
    headers  = read_tsv_headers(SCHEMA_TSV)
    mapping  = annotate_to_json(text, tokenizer, model)
    spans    = build_spans(mapping, text)
    stem     = Path(TXT_PATH).stem
    out_file = os.path.join(OUTPUT_DIR, f"{stem}_annotations.tsv")
    lines    = json_to_webanno(spans, headers, text)
    save_tsv(lines, out_file)
    logger.info("Inference complete.")
