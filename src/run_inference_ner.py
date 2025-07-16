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
from peft import PeftModelForCausalLM

logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s: %(asctime)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 1. UPDATED PROMPT 
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

Respond with a JSON dictionary mapping each entity to its label, e.g.: {{"Entity": "Label", ...}}

### Input
{text}

Entities:
"""

# ─────────────────────────────────────────────────────────────────────────────
def load_model(model_name: str):
    logger.info("Attempting to load model: %s", model_name)
    # Determine if model_name is a local PEFT adapter directory
    is_adapter = os.path.isdir(model_name)
    base_model_name = 'Qwen/Qwen2.5-14B-Instruct'

    # Load tokenizer from the base model (always full tokenizer)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        logger.info("Loaded tokenizer from: %s", base_model_name)
    except Exception as e:
        logger.error("Failed to load tokenizer from base model: %s", e)
        raise

    # Load base model
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        logger.info("Loaded base model: %s", base_model_name)
    except Exception as e:
        logger.error("Failed to load base model: %s", e)
        raise

    # If adapter directory provided, attach it
    if is_adapter:
        try:
            model = PeftModelForCausalLM.from_pretrained(
                base_model,
                model_name,
                trust_remote_code=True,
                device_map="auto"
            )
            logger.info("Loaded PEFT adapter from: %s", model_name)
        except Exception as e:
            logger.error("Failed to load PEFT adapter from '%s': %s", model_name, e)
            raise
    else:
        model = base_model
        logger.info("Using base model without adapter.")

    return tokenizer, model


# ─────────────────────────────────────────────────────────────────────────────
def read_tsv_headers(tsv_path: str) -> List[str]:
    """
    Read WebAnno TSV headers to get the list of entity labels (in order).
    """
    logger.info("Reading WebAnno TSV headers from: %s", tsv_path)
    headers: List[str] = []
    try:
        with open(tsv_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('#T_SP=webanno.custom.'):
                    label = line.strip().split('.')[-1].split('|')[0]
                    headers.append(label)
        if not headers:
            logger.warning("No entity headers found in TSV schema: %s", tsv_path)
        logger.info("Found %d entity types: %s", len(headers), headers)
    except FileNotFoundError:
        logger.error("TSV schema file not found: %s", tsv_path)
        raise
    except Exception as e:
        logger.error("Error reading TSV headers from %s: %s", tsv_path, e)
        raise
    return headers

# ─────────────────────────────────────────────────────────────────────────────
def read_text(text_path: str) -> str:
    """
    Read the input text file, attempting UTF-8 → Latin-1 → replacement.
    Normalize unicode (NFKC) and collapse whitespace.
    """
    logger.info("Reading input text from: %s", text_path)
    try:
        raw = Path(text_path).read_bytes()
        try:
            text = raw.decode('utf-8')
            logger.debug("Decoded as UTF-8")
        except UnicodeDecodeError:
            logger.warning("UTF-8 decode failed for %s; trying latin-1.", text_path)
            try:
                text = raw.decode('latin-1')
                logger.debug("Decoded as latin-1")
            except Exception as e2:
                logger.error("Latin-1 decode failed for %s: %s; falling back to utf-8 with replacement.", text_path, e2)
                text = raw.decode('utf-8', errors='replace')
        # Normalize & collapse whitespace AFTER decoding
        # NFKC is used to handle compatibility characters, useful for matching
        text = unicodedata.normalize('NFKC', text)
        # Collapse all whitespace sequences (including newlines, tabs) to a single space
        text = re.sub(r"\s+", " ", text).strip()
        logger.info("Completed read_text, length %d chars after normalization. First 100 chars: '%s...'", len(text), text[:100])
        return text
    except FileNotFoundError:
        logger.error("Input text file not found: %s", text_path)
        raise
    except Exception as e:
        logger.error("Error reading text from %s: %s", text_path, e)
        raise

# ─────────────────────────────────────────────────────────────────────────────
def extract_single_json(response: str) -> Dict[str, str]:
    """
    Extract exactly one JSON object from the model's response string.
    This version is more robust, looking for the last JSON object.
    It also tries to clean up markdown fences if present.
    """
    response = response.strip()
    logger.debug("Attempting to extract JSON from response (length %d):\n%s", len(response), response[:1000] + ('...' if len(response) > 1000 else ''))

    # Clean up common markdown fences that models might mistakenly include
    # Use a regex to find JSON blocks, accounting for optional markdown fences
    json_pattern = re.compile(r"```(?:json)?\s*({.*})\s*```", re.DOTALL)
    match = json_pattern.search(response)
    if match:
        json_str = match.group(1)
        logger.debug("Found JSON within markdown fences.")
    else:
        # Fallback to finding the first '{' and last '}'
        first_open = response.find("{")
        last_close = response.rfind("}")

        if first_open == -1 or last_close == -1 or last_close < first_open:
            logger.error("No valid JSON braces found in response.")
            return {}
        json_str = response[first_open : last_close + 1]
        logger.debug("Found JSON by brace matching.")

    try:
        data = json.loads(json_str)
        if not isinstance(data, dict):
            logger.warning("Parsed JSON is not a dictionary. Expected a flat JSON object.")
            return {}
        logger.info("Successfully parsed JSON map with %d entries.", len(data))
        return data
    except json.JSONDecodeError as e:
        logger.error("JSON parse failed for extracted string:\n---\n%s\n---\nError: %s", json_str, e)
        return {}
    except Exception as e:
        logger.error("Unexpected error during JSON extraction: %s", e)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# 8192, 16384 
def annotate_to_json(text: str, tokenizer, model, max_tokens:int = 16384) -> Dict[str, str]:
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
def normalize_phrase_for_matching(phrase: str) -> str:
    """
    - NFKC then NFKD Unicode normalize
    - Remove diacritics
    - Replace underscores with spaces
    - Collapse whitespace
    - Lowercase
    """
    text = unicodedata.normalize('NFKC', str(phrase))
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')
    text = text.replace('_', ' ')
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

# ─────────────────────────────────────────────────────────────────────────────
def build_spans(span_map: Dict[str,str], original_text: str) -> List[Tuple[int,int,str]]:
    spans: List[Tuple[int,int,str]] = []
    norm_text = normalize_phrase_for_matching(original_text)
    for phrase, label in span_map.items():
        norm_phrase = normalize_phrase_for_matching(phrase)
        found = False
        pattern = re.escape(norm_phrase)
        for m in re.finditer(pattern, norm_text, flags=re.IGNORECASE):
            spans.append((m.start(), m.end(), label))
            found = True
        if not found:
            logger.warning("No match: '%s' orig:'%s'", norm_phrase, phrase)
    spans.sort(key=lambda x: x[0])
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
    This version includes better handling of span-to-token mapping and active spans.
    """
    logger.info("Converting spans to WebAnno TSV format using spaCy...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logger.error("spaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
        raise
    doc = nlp(text)
    sentences = list(doc.sents)
    logger.info("Text split into %d sentences by spaCy.", len(sentences))

    lines: List[str] = ["#FORMAT=WebAnno TSV 3.3"]
    for h in headers:
        lines.append(f"#T_SP=webanno.custom.{h}|")

    annotation_id_counter = 1
    # Keep track of spans that are currently active over tokens
    # Key: span_tuple (start, end, label), Value: WebAnno annotation ID
    active_span_annotations: Dict[Tuple[int, int, str], int] = {}
    
    # Store token-level annotations temporarily before writing line
    token_annotations: List[List[Tuple[int, str]]] = [] # List of (annotation_id, label_header_index) tuples per token

    for sent_idx, sent in enumerate(sentences, start=1):
        lines.append(f"\n#Text={sent.text}")
        
        # Reset active spans for a new sentence if needed, or maintain across sentences for multi-sentence entities (less common)
        # For simplicity and typical NER, spans are often sentence-bound.
        # If an entity spans multiple sentences, the current `build_spans` and `json_to_webanno` may need adjustment.
        # Current logic handles tokens crossing sentence boundaries if the span covers it.

        for tok_idx, tok in enumerate(sent, start=1):
            token_text = tok.text
            start_char = tok.idx
            end_char = start_char + len(token_text)

            current_token_labels: List[Tuple[int, str]] = [] # (annotation_id, header_idx) for this token

            # Find all spans that fully contain or are fully contained by the token (or overlap significantly)
            # More robust check: if token is *within* a span.
            # We prioritize exact match as per the prompt.
            matching_spans_for_token = []
            for s_start, s_end, s_label in spans:
                # A token is considered "covered" if its start is within the span and its end is within the span
                # Or if the span starts within the token and ends within the token (token contains span)
                # Or if the span completely contains the token
                if (start_char >= s_start and end_char <= s_end) or \
                   (s_start >= start_char and s_end <= end_char) or \
                   (start_char < s_end and end_char > s_start): # Any overlap
                    
                    # For WebAnno, we want to assign a label if the token is *part of* the entity.
                    # The most straightforward way is if the token's boundaries are fully within the span,
                    # or if the span's boundaries are fully within the token (for single-token entities).
                    # A more nuanced approach: if the token is *partially* inside, assign it.
                    # For simplicity, let's say a token is part of a span if it *overlaps*.
                    
                    # A common heuristic is: if a token's start or end is within a span, or span is within token
                    if (s_start <= start_char < s_end) or \
                       (s_start < end_char <= s_end) or \
                       (start_char <= s_start and end_char >= s_end):
                        matching_spans_for_token.append((s_start, s_end, s_label))
            
            # Sort matching spans by length (longest first) to prioritize larger entities
            # Or by start position for consistency
            matching_spans_for_token.sort(key=lambda x: (x[0], -(x[1]-x[0]))) # Sort by start, then by length (desc)

            for s_start, s_end, s_label in matching_spans_for_token:
                span_key = (s_start, s_end, s_label)

                # Assign a new annotation ID if this span hasn't been started yet for WebAnno
                if span_key not in active_span_annotations:
                    active_span_annotations[span_key] = annotation_id_counter
                    annotation_id_counter += 1
                
                # Get the WebAnno ID for this span
                ann_id = active_span_annotations[span_key]
                
                # Find the index of the label in the headers list
                try:
                    header_idx = headers.index(s_label)
                    current_token_labels.append((ann_id, header_idx))
                except ValueError:
                    logger.warning("Label '%s' for span [%d,%d] not found in TSV headers. Skipping.", s_label, s_start, s_end)
            
            # Now, construct the TSV row
            row = [f"{sent_idx}-{tok_idx}", f"{start_char}-{end_char}", token_text]
            
            # Create a list for each header column, default to "_"
            header_columns_data = ["_"] * len(headers)
            
            # Populate header columns based on active token labels
            # If multiple labels apply to one token, WebAnno puts them like `*[1];*[2]`
            # However, typically for NER, one token gets one label type.
            # Your current setup doesn't handle overlapping labels for a single token in TSV 3.3 by default.
            # For simplicity, if multiple labels are found for a token, we might just take the first.
            
            # Store labels for the token, handling potential duplicates for the same label type
            temp_labels_per_header: Dict[int, List[int]] = {idx: [] for idx, _ in enumerate(headers)}
            
            for ann_id, header_idx in current_token_labels:
                temp_labels_per_header[header_idx].append(ann_id)
            
            for h_idx, _ in enumerate(headers):
                if temp_labels_per_header[h_idx]:
                    # Format for WebAnno is `*[ID]` or `*[ID];*[ID2]` for multi-layer/overlapping
                    # Given your task, usually one label per token.
                    # Let's assume the first match found is the primary for the column.
                    formatted_anns = ";".join([f"*[{aid}]" for aid in sorted(temp_labels_per_header[h_idx])])
                    header_columns_data[h_idx] = formatted_anns

            row.extend(header_columns_data)
            lines.append("\t".join(row))

    logger.info("WebAnno TSV conversion completed. Total annotation spans created: %d.", annotation_id_counter - 1)
    return lines

# ─────────────────────────────────────────────────────────────────────────────
def save_tsv(lines: List[str], out_path: str):
    """
    Save lines as a WebAnno TSV 3.3 file. Overwrites if already present.
    """
    logger.info("Saving WebAnno TSV to: %s", out_path)
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info("WebAnno TSV save complete.")
    except Exception as e:
        logger.error("Error saving TSV to %s: %s", out_path, e)
        raise

# ─────────────────────────────────────────────────────────────────────────────

def process_pair(tsv_schema: str, text_file: str, output_file: str, tokenizer, model):
    """
    Process one schema-text pair: run annotation → JSON → spans → WebAnno TSV → save.
    """
    logger.info("Starting processing for pair: schema='%s', text='%s', output='%s'", tsv_schema, text_file, output_file)
    try:
        headers = read_tsv_headers(tsv_schema)
        text = read_text(text_file)
        token_map = annotate_to_json(text, tokenizer, model)
        
        if not token_map:
            logger.error("No entities extracted from model. Skipping span building and TSV conversion.")
            return

        # --- NEW CODE START ---
        # Define the path for the JSON output file
        json_output_file = output_file.replace(".tsv", ".json")
        logger.info("Saving extracted JSON to: %s", json_output_file)
        try:
            with open(json_output_file, "w", encoding="utf-8") as f:
                json.dump(token_map, f, ensure_ascii=False, indent=4)
            logger.info("Extracted JSON saved successfully.")
        except Exception as e:
            logger.error("Error saving extracted JSON to %s: %s", json_output_file, e)
        # --- NEW CODE END ---

        spans = build_spans(token_map, text)
        
        if not spans:
            logger.warning("No spans could be built from the extracted entities. Output TSV will be empty of annotations.")

        webanno_lines = json_to_webanno(spans, headers, text)
        save_tsv(webanno_lines, output_file)
        logger.info("Successfully processed pair: %s", output_file)
    except Exception as e:
        logger.critical("Critical error during processing of pair (schema: %s, text: %s): %s", tsv_schema, text_file, e)

# ─────────────────────────────────────────────────────────────────────────────
if __name__=='__main__':
    TSV_DIR   = 'data/few-shot/(1) Axelsson 2023.tsv'
    TXT_DIR   = 'data/test-copy/(1) Axelsson 2023.txt'
    OUT_DIR   = 'output-finetuned1'
    MODEL_NAME = "./qwen_ner_instruction_finetuned_v2"

    tokenizer, model = load_model(MODEL_NAME)
    stem = Path(TXT_DIR).stem
    out_path = os.path.join(OUT_DIR, f"{stem}_annotations.tsv")
    process_pair(TSV_DIR, TXT_DIR, out_path, tokenizer, model)
    logger.info("Complete processing for single pair: %s", out_path)

