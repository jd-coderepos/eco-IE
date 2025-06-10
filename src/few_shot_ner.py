import os
import re
import spacy
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import unicodedata

logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s: %(asctime)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 1. UPDATED PROMPT to EXCLUDE PEOPLE/AUTHORS from entities
SYSTEM_PROMPT = "You are an expert in scientific information extraction."

FEW_SHOT_PROMPT ="""## Task Description
Perform Named Entity Recognition (NER) for scientific text. For each entity, assign one of the following labels.

1. "Timeperiodofstudy": Refers to information about the timing of the study, including beginning and end date, total duration, timing, and duration of fieldwork. Usually found in the Abstract/Introduction/Methods sections. (e.g., December 2024)
2. "Locationofstudy": Refers to information about the physical setting, coordinates, geographical location, and name of country/city. Usually found in the Abstract/Introduction/Methods sections. (e.g., New York, plains of America)
3. "Ecosystem": Refers to information describing the ecosystem of the study or where the species is present, including type of ecosystem, land use history, abiotic conditions, and sometimes location with descriptors. Usually found in the Methods sections. (e.g., Temperature, terrain, precipitation, nutrients, plains of America)
4. "Focalpoint": Refers to the species discussed in the study. (e.g., Tulipa gesneriana)
5. "Method": Refers to the type of study and its conditions, including whether it is an experiment, fieldwork, or survey, and the number/types of samples and variables measured. (e.g., Collecting 50 samples, regression analysis, using GIS)
6. "Researchquestions": Refers to the problems or questions addressed, including research gaps. Usually found in the last part of the Introduction/Method sections. (e.g., “We performed this study to determine whether X impacts Y”)
7. "Mainhypothesisandcorrespondingresults": Refers to the hypotheses and theories posed by the authors and the main outcomes/results of the study. Usually found in the Results/Discussion sections. (e.g., “Habitat X had the highest amount of species mortality”)
8. "Causalstatements": Refers to statements that clearly depict a cause-effect relationship. These do not have a designated location and may not appear in every text. (e.g., “The absence of species X resulted in the growth in species Y”)
9. "Reccomendationsandsuggestions": Refers to statements not backed by evidence or results, based on the authors' conclusions or knowledge. Also includes suggestions to alter methods. Usually found in the Discussion section.

## Instructions
- Match entities exactly as they appear in the text, including punctuation and multi-word tokens.
- Use only the provided entity labels.
- Do not include explanations, headers, or additional text.
- Do NOT label any person, author, research group, organization, or any list of names as an entity.

Example 1:
Text: "Collecting 50 samples from a designated area in December 2024 allowed us to study Tulipa gesneriana in the plains of America."
Entities: {{"Collecting 50 samples from a designated area": "Method", "December 2024": "Timeperiodofstudy", "Tulipa gesneriana": "Focalpoint", "plains of America": "Locationofstudy"}}

Example 2:
Text: "We performed this study to determine whether precipitation influences temperature in New York."
Entities: {{"We performed this study to determine whether precipitation influences temperature": "Researchquestions", "precipitation": "Ecosystem", "temperature": "Ecosystem", "New York": "Locationofstudy"}}

Respond with a JSON dictionary mapping each entity to its label, e.g.: {{"Entity": "Label", ...}}

### Input
{text}

Entities:
"""

# ─────────────────────────────────────────────────────────────────────────────
def load_model(model_name: str):
    logger.info("torch.cuda.is_available(): %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("CUDA device count: %d", torch.cuda.device_count())
        logger.info("CUDA current device: %s", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        logger.warning("CUDA not available—model will load on CPU.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    logger.info("Model parameters device after load: %s", next(model.parameters()).device)
    return tokenizer, model

# ─────────────────────────────────────────────────────────────────────────────
def read_tsv_headers(tsv_path: str) -> List[str]:
    """
    Read WebAnno TSV headers to get the list of entity labels.
    """
    logger.info("Reading WebAnno TSV headers from: %s", tsv_path)
    headers: List[str] = []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#T_SP=webanno.custom.'):
                label = line.strip().split('.')[-1].split('|')[0]
                headers.append(label)
    logger.info("Found %d entity types: %s", len(headers), headers)
    return headers

# ─────────────────────────────────────────────────────────────────────────────
def read_text(text_path: str) -> str:
    """
    Read the input text file with fallback encodings to handle non-UTF8 bytes.
    Normalize unicode and collapse whitespace.
    """
    logger.info("Reading input text from: %s", text_path)
    path = Path(text_path)
    raw = path.read_bytes()
    try:
        text = raw.decode('utf-8')
        logger.info("Decoded text as UTF-8.")
    except UnicodeDecodeError as e:
        logger.warning("UTF-8 decode failed: %s. Trying latin-1.", e)
        try:
            text = raw.decode('latin-1')
            logger.info("Decoded text as latin-1.")
        except Exception as e2:
            logger.error("Failed to decode text as latin-1: %s", e2)
            text = raw.decode('utf-8', errors='replace')
            logger.info("Decoded text with utf-8 replace errors.")

    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    logger.info("Completed read_text, normalized length %d chars", len(text))
    return text

# ─────────────────────────────────────────────────────────────────────────────

def extract_single_json(response: str) -> Dict[str, str]:
    """
    Extract exactly one JSON object from the model response string.
    Finds the first '{' and the matching last '}', then attempts to parse that slice.
    """
    response = response.strip()
    first_open = response.find("{")
    last_close = response.rfind("}")
    if first_open == -1 or last_close == -1 or last_close < first_open:
        logger.error("No valid outer braces found in response. Full response:\n%s", response)
        return {}

    json_str = response[first_open : last_close + 1]
    try:
        data = json.loads(json_str)
        logger.info("Parsed JSON map with %d entries", len(data))
        return data
    except json.JSONDecodeError as e:
        logger.error("JSON decode failed on substring:\n%s\nError: %s", json_str, e)
        return {}

# ─────────────────────────────────────────────────────────────────────────────
def annotate_to_json(text: str, tokenizer, model, max_tokens:int = 2048) -> Dict[str, str]:
    """
    Annotate the text via the Instruct LLM to get a JSON map: phrase -> label.
    Uses tokenizer.apply_chat_template() and model.generate(), then extracts JSON.
    """
    logger.info("Starting annotation via LLM...")
    user_prompt = FEW_SHOT_PROMPT.format(text=text)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt}
    ]
    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([chat_prompt], return_tensors='pt').to(model.device)

    # Generate up to max_tokens additional tokens
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    # Remove the prompt tokens so we only decode the model’s response
    gen_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)]
    response = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
    logger.debug("Raw model response:\n%s", response)

    token_map = extract_single_json(response)
    if not token_map:
        logger.error("Failed to extract JSON. No tokens to annotate.")
    else:
        logger.debug("Parsed token-label map: %s", json.dumps(token_map, indent=2))
    return token_map

# ─────────────────────────────────────────────────────────────────────────────
def build_spans(span_map: Dict[str, str], text: str) -> List[Tuple[int,int,str]]:
    """
    Convert a phrase->label map into character-offset spans over the original text.
    Each phrase is normalized (NFKC + collapse whitespace), found in normalized text,
    then located in the original text via .find() to compute exact offsets.
    """
    logger.info("Entered build_spans with %d items", len(span_map))
    spans: List[Tuple[int,int,str]] = []
    normalized_text = unicodedata.normalize("NFKC", text)
    normalized_text = re.sub(r"\s+", " ", normalized_text)

    for phrase, label in span_map.items():
        norm_phrase = unicodedata.normalize("NFKC", phrase)
        norm_phrase = re.sub(r"\s+", " ", norm_phrase)

        for m in re.finditer(re.escape(norm_phrase), normalized_text, flags=re.IGNORECASE):
            orig_start = text.find(phrase)
            if orig_start == -1:
                logger.warning("Phrase not found literally in original text: '%s'", phrase)
                continue
            orig_end = orig_start + len(phrase)
            spans.append((orig_start, orig_end, label))
            logger.debug(
                "Matched phrase '%s' → offsets [%d,%d], label=%s",
                phrase, orig_start, orig_end, label
            )

    spans.sort(key=lambda x: x[0])
    logger.info("Built %d spans from JSON", len(spans))
    return spans

# ─────────────────────────────────────────────────────────────────────────────
def json_to_webanno(
    spans: List[Tuple[int,int,str]],
    headers: List[str],
    text: str
) -> List[str]:
    """
    Convert character-offset spans and headers into WebAnno TSV 3.3 lines.
    Each spaCy token is checked for coverage by any span, and labeled accordingly.
    """
    logger.info("Converting spans to WebAnno TSV format")
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    sentences = list(doc.sents)
    logger.info("Document split into %d sentences", len(sentences))

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

            # Determine label by checking if the token falls inside any span
            label = 'O'
            for (s0, s1, span_label) in spans:
                if start_char >= s0 and end_char <= s1:
                    label = span_label
                    break

            # Build the TSV row
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

# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def save_tsv(lines: List[str], out_path: str):
    """
    Save the list of lines as a WebAnno TSV 3.3 file.
    If the output already exists, skip saving.
    """
    if Path(out_path).exists():
        logger.info("Output already exists, skipping: %s", out_path)
        return
    logger.info("Saving WebAnno TSV to: %s", out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    logger.info("WebAnno TSV successfully saved")

# ─────────────────────────────────────────────────────────────────────────────
def process_pair(tsv_schema: str, text_file: str, output_file: str, tokenizer, model):
    """
    Process one schema-text pair: run annotation → JSON → spans → WebAnno TSV → save.
    """
    logger.info("Processing pair: schema=%s, text=%s, output=%s", tsv_schema, text_file, output_file)
    headers = read_tsv_headers(tsv_schema)
    text = read_text(text_file)
    token_map = annotate_to_json(text, tokenizer, model)
    spans = build_spans(token_map, text)
    webanno_lines = json_to_webanno(spans, headers, text)
    save_tsv(webanno_lines, output_file)

# ─────────────────────────────────────────────────────────────────────────────
def batch_process(
    tsv_dir: str,
    txt_dir: str,
    out_dir: str,
    model_name: str
):
    """
    Batch-process all .tsv schemas and .txt files in their respective directories.
    Creates output for every possible .txt × .tsv pair, naming files as:
      {txt_stem}__{tsv_stem}_annotations.tsv
    Skips any output that already exists.
    """
    logger.info("Starting batch processing: schemas=%s, texts=%s, out=%s", tsv_dir, txt_dir, out_dir)

    # Collect all schema and text files
    schema_paths = list(Path(tsv_dir).glob("*.tsv"))
    text_paths   = list(Path(txt_dir).glob("*.txt"))

    if not schema_paths:
        logger.warning("No .tsv files found in %s", tsv_dir)
    if not text_paths:
        logger.warning("No .txt files found in %s", txt_dir)

    # Load model once
    tokenizer, model = load_model(model_name)

    # For each combination of schema & text
    for schema_path in schema_paths:
        tsv_stem = schema_path.stem
        for text_path in text_paths:
            txt_stem = text_path.stem
            # Output filename: txtstem__tsvstem_annotations.tsv
            output_filename = f"{txt_stem}__{tsv_stem}_annotations.tsv"
            output_file = os.path.join(out_dir, output_filename)

            if Path(output_file).exists():
                logger.info("Output already exists for '%s' × '%s', skipping", txt_stem, tsv_stem)
                continue

            process_pair(str(schema_path), str(text_path), output_file, tokenizer, model)

    logger.info("Batch processing complete")

# ─────────────────────────────────────────────────────────────────────────────
if __name__=='__main__':
    TSV_DIR   = 'data/train'
    TXT_DIR   = 'data/test-copy'
    OUT_DIR   = 'output/few_shot'
    MODEL_NAME = 'Qwen/Qwen2.5-14B-Instruct'

    batch_process(TSV_DIR, TXT_DIR, OUT_DIR, MODEL_NAME)

