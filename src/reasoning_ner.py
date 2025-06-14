# qwen3_reasoning_ner.py – Single-file NER pipeline using Qwen/Qwen3-1.7B reasoning model
# with "think step by step" and WebAnno TSV postprocessing.

import os
import re
import json
import unicodedata
import logging
import spacy
import torch

from pathlib import Path
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s: %(asctime)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Prompts
SYSTEM_PROMPT = "You are an expert in scientific reasoning and information extraction."
USER_CONTEXT = """## Task Description
Perform step-by-step reasoning to identify Named Entities in the scientific text.  At the very end, output exactly one JSON object.

Each JSON key must be a single, exact substring from the input text.  Each JSON value must be exactly one of these nine labels (no spelling variants):

1. "Timeperiodofstudy": Refers to information about the timing of the study, including beginning and end date, total duration, timing, and duration of fieldwork. Usually found in the Abstract/Introduction/Methods sections.
2. "Locationofstudy": Refers to information about the physical setting, coordinates, geographical location, and name of country/city. Usually found in the Abstract/Introduction/Methods sections.
3. "Ecosystem": Refers to information describing the ecosystem of the study or where the species is present, including type of ecosystem, land use history, abiotic conditions, and sometimes location with descriptors. Usually found in the Methods sections.
4. "Focalpoint": Refers to the species discussed in the study.
5. "Method": Refers to the type of study and its conditions, including whether it is an experiment, fieldwork, or survey, and the number/types of samples and variables measured.
6. "Researchquestions": Refers to the problems or questions addressed, including research gaps. Usually found in the last part of the Introduction/Method sections.
7. "Mainhypothesisandcorrespondingresults": Refers to the hypotheses and theories posed by the authors and the main outcomes/results of the study. Usually found in the Results/Discussion sections.
8. "Causalstatements": Refers to statements that clearly depict a cause-effect relationship. These do not have a designated location and may not appear in every text.
9. "Reccomendationsandsuggestions": Refers to statements not backed by evidence or results, based on the authors' conclusions or knowledge. Also includes suggestions to alter methods. Usually found in the Discussion section.

## Instructions
- If the text contains multiple distinct phrases all belonging to the same label (for example, four different entities under “Focalpoint”), you must emit each phrase as its own JSON key.  
- Never group more than one phrase under a single key.  
- The output must be a single, flat JSON dictionary.  No lists or nested objects.  
- Do not output any extra text, no commentary, no markdown—just the JSON.

### Input
{text}

### Output
A single JSON dictionary mapping each exact entity phrase to its correct label.  Ensure one key→one value per phrase, with no additional lines.
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
    Read WebAnno TSV headers to get the list of entity labels (in order).
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
    Read the input text file, attempting UTF-8 → Latin-1 → replacement.
    Normalize unicode (NFKC) and collapse whitespace.
    """
    logger.info("Reading input text from: %s", text_path)
    raw = Path(text_path).read_bytes()
    try:
        text = raw.decode('utf-8')
        logger.info("Decoded as UTF-8")
    except UnicodeDecodeError as e:
        logger.warning("UTF-8 decode failed: %s; trying latin-1", e)
        try:
            text = raw.decode('latin-1')
            logger.info("Decoded as latin-1")
        except Exception as e2:
            logger.error("Latin-1 decode failed: %s; using utf-8 replace", e2)
            text = raw.decode('utf-8', errors='replace')
    # Normalize & collapse whitespace
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r"\s+", " ", text).strip()
    logger.info("Completed read_text, length %d chars", len(text))
    return text

# ─────────────────────────────────────────────────────────────────────────────
def extract_single_json(response: str) -> Dict[str, str]:
    """
    Extract exactly one JSON object from the model's response string.
    Finds the first '{' and last '}', then attempts json.loads().
    """
    response = response.strip()
    first_open = response.find("{")
    last_close = response.rfind("}")
    if first_open == -1 or last_close == -1 or last_close < first_open:
        logger.error("No valid JSON braces found. Full response:\n%s", response)
        return {}

    json_str = response[first_open : last_close + 1]
    try:
        data = json.loads(json_str)
        logger.info("Parsed JSON map with %d entries", len(data))
        return data
    except json.JSONDecodeError as e:
        logger.error("JSON parse failed:\n%s\nError: %s", json_str, e)
        return {}

# ─────────────────────────────────────────────────────────────────────────────

def annotate_to_json(text: str, tokenizer, model, max_tokens: int = 32768) -> Dict[str, str]:
    """
    Annotate text via Qwen3-1.7B reasoning model. Uses apply_chat_template with enable_thinking=True.
    Returns a dict: { "Entity phrase": "Label", ... }.
    """
    logger.info("Starting reasoning-based annotation...")
    user_prompt = USER_CONTEXT.format(text=text)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt}
    ]
    # Build chat prompt with thinking enabled
    chat_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    inputs = tokenizer([chat_input], return_tensors="pt").to(model.device)
    # Generate up to max_tokens additional tokens
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    # Strip off prompt tokens, keep model response
    gen_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)]
    # Only one sequence
    full_output = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    # The reasoning model interleaves <think>…</think> and then the final content
    # We'll locate the final content after the last '</think>'
    think_end_token_id = 151668  # per Qwen3-1.7B def
    token_ids = gen_ids[0].tolist()
    try:
        # Find the last occurrence of the token ID for '</think>'
        idx = len(token_ids) - token_ids[::-1].index(think_end_token_id)
    except ValueError:
        idx = 0

    thinking_ids = token_ids[:idx]
    content_ids  = token_ids[idx:]
    thinking_content = tokenizer.decode(thinking_ids,  skip_special_tokens=True).strip()
    final_content    = tokenizer.decode(content_ids, skip_special_tokens=True).strip()

    logger.debug("Thinking content:\n%s", thinking_content)
    logger.debug("Final JSON content:\n%s", final_content)

    # Extract JSON from final_content
    token_map = extract_single_json(final_content)
    return token_map

# ─────────────────────────────────────────────────────────────────────────────
def build_spans(span_map: Dict[str, str], text: str) -> List[Tuple[int,int,str]]:
    """
    Convert phrase→label map into character-offset spans over the original text.
    Normalizes both phrase and text (NFKC + collapse whitespace), finds matches,
    then locates the phrase literally in the original text to compute offsets.
    """
    logger.info("Building spans from %d phrases", len(span_map))
    # Step 1: Build mapping from normalized positions → original positions
    normalized_chars = []
    norm_to_orig = []
    for orig_idx, ch in enumerate(text):
        if ch.isspace():
            # collapse any run of whitespace to a single space
            if normalized_chars and normalized_chars[-1] == " ":
                continue
            normalized_chars.append(" ")
            norm_to_orig.append(orig_idx)
        else:
            # lowercase the character
            normalized_chars.append(ch.lower())
            norm_to_orig.append(orig_idx)

    normalized_text = "".join(normalized_chars)

    spans: List[Tuple[int,int,str]] = []
    for phrase, label in span_map.items():
        # Step 2: normalize the phrase
        norm_phrase = unicodedata.normalize("NFKC", phrase).lower()
        norm_phrase = re.sub(r"\s+", " ", norm_phrase).strip()

        # Find every match of norm_phrase in normalized_text
        for m in re.finditer(re.escape(norm_phrase), normalized_text):
            norm_start, norm_end = m.start(), m.end()
            # Map back to original indices:
            orig_start = norm_to_orig[norm_start]
            # norm_end - 1 maps to last normalized char, so add 1 to get exclusive end
            orig_end   = norm_to_orig[norm_end - 1] + 1
            spans.append((orig_start, orig_end, label))
            logger.debug("Loose‐matched '%s' → [%d,%d] as %s",
                         phrase, orig_start, orig_end, label)

    spans.sort(key=lambda x: x[0])
    logger.info("Built %d spans (loose match)", len(spans))
    return spans

# ─────────────────────────────────────────────────────────────────────────────


def json_to_webanno(
    spans: List[Tuple[int,int,str]],
    headers: List[str],
    text: str
) -> List[str]:
    """
    Convert character-offset spans + headers into WebAnno TSV 3.3 lines.
    Uses spaCy to tokenize into sentences/tokens; assigns labels by span coverage.
    """
    logger.info("Converting spans to WebAnno TSV format")
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    sentences = list(doc.sents)
    logger.info("Text split into %d sentences", len(sentences))

    lines: List[str] = ["#FORMAT=WebAnno TSV 3.3"]
    for h in headers:
        lines.append(f"#T_SP=webanno.custom.{h}|")

    ann_counter = 1
    active_spans: Dict[str,int] = {}
    prev_label: str = None

    for sent_idx, sent in enumerate(sentences, start=1):
        lines.append(f"\n#Text={sent.text}")
        for tok_idx, tok in enumerate(sent, start=1):
            token_text = tok.text
            start_char = tok.idx
            end_char = start_char + len(token_text)

            # Determine token's label by checking if it falls inside any span
            label = "O"
            for (s0, s1, span_label) in spans:
                if start_char >= s0 and end_char <= s1:
                    label = span_label
                    break

            row = [f"{sent_idx}-{tok_idx}", f"{start_char}-{end_char}", token_text]
            for h in headers:
                if label == h:
                    if label != prev_label:
                        active_spans[label] = ann_counter
                        ann_counter += 1
                    row.append(f"*[{active_spans[label]}]")
                else:
                    row.append("_")
            prev_label = label if label != "O" else None
            lines.append("\t".join(row))

    logger.info("WSV conversion done with %d annotation spans", ann_counter - 1)
    return lines

# ─────────────────────────────────────────────────────────────────────────────
def save_tsv(lines: List[str], out_path: str):
    """
    Save lines as a WebAnno TSV 3.3 file. Overwrites if already present.
    """
    logger.info("Saving WebAnno TSV to: %s", out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Save complete")

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
    TXT_DIR   = 'data/test'
    OUT_DIR   = 'output/reasoning_ner'
    MODEL_NAME = 'Qwen/Qwen3-14B'

    batch_process(TSV_DIR, TXT_DIR, OUT_DIR, MODEL_NAME)
