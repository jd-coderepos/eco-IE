import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import json
import re
import logging
import os # For directory creation and file paths
import spacy # New: Import spaCy
from typing import List, Tuple, Dict # For type hints in the new function

# --- Configure Logging ---
# Set logging level (DEBUG for most verbose, INFO for general progress)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct" # Change to "Qwen/Qwen2-14B-Instruct" for the larger model
OUTPUT_DIR = "./qwen_ner_instruction_finetuned_v2"
MAX_SEQ_LENGTH = 1024 # Increased for longer instruction/response pairs
GENERATION_MAX_NEW_TOKENS = 1024 # Increased for more complete JSON generation

# Define your NER labels. This list will now be dynamically loaded from a TSV file.
# Keeping it here as a fallback or for initial definition if no TSV is provided.
DEFAULT_NER_LABELS = [
    "Ecosystem", "Focalpoint","Causalstatements", "Locationofstudy",
    "Mainhypothesisandcorrespondingresults", "Method",
    "Reccomendationsandsuggestions", "Researchquestions", "Timeperiodofstudy"
]
logging.info(f"Default NER Entity Types: {DEFAULT_NER_LABELS}")

# --- Load spaCy model globally or within function with caching ---
_nlp_model = None
try:
    # Attempt to load the model. If not found, guide the user to download.
    _nlp_model = spacy.load("en_core_web_sm")
    logging.info("spaCy model 'en_core_web_sm' loaded successfully.")
except OSError:
    logging.error("spaCy model 'en_core_web_sm' not found. Please run: 'pip install spacy' and then 'python -m spacy download en_core_web_sm'")
    logging.error("WebAnno TSV conversion will not work without spaCy model.")


# --- Helper Function to Read WebAnno Headers ---
def read_webanno_headers(tsv_file_path):
    """
    Reads entity type declarations from a WebAnno TSV file header.
    Expected format: #T_SP=webanno.custom.EntityType|
    Returns a list of entity type strings.
    """
    logging.info(f"Attempting to read WebAnno headers from: {tsv_file_path}")
    ner_labels = []
    try:
        with open(tsv_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith("#T_SP="):
                    # Extract "EntityType" from "#T_SP=webanno.custom.EntityType|"
                    match = re.search(r"#T_SP=webanno\.custom\.([^|]+)\|", line)
                    if match:
                        entity_type = match.group(1)
                        ner_labels.append(entity_type)
                        logging.debug(f"  Found entity type in header: {entity_type}")
                elif not line.strip() and ner_labels: # Stop after header block if labels found
                    break
                elif not line.startswith("#") and ner_labels: # Stop if we hit content after header
                    break
        if not ner_labels:
            logging.warning(f"No #T_SP= lines found in {tsv_file_path}. Using default NER labels.")
            return DEFAULT_NER_LABELS
        logging.info(f"Successfully read {len(ner_labels)} entity types from {tsv_file_path}.")
        return ner_labels
    except FileNotFoundError:
        logging.error(f"Error: TSV header file '{tsv_file_path}' not found. Using default NER labels.")
        return DEFAULT_NER_LABELS
    except Exception as e:
        logging.error(f"Error reading TSV header file '{tsv_file_path}': {e}. Using default NER labels.")
        return DEFAULT_NER_LABELS # Corrected variable name


# --- New Function: Read WebAnno TSV to BIO format for fine-tuning ---
def read_webanno_tsv_to_bio(tsv_file_path):
    logging.info(f"Reading WebAnno TSV file '{tsv_file_path}' for BIO conversion.")
    bio_data = []
    current_tokens = []
    current_labels = []
    
    # Store {span_id: entity_type} for spans that are currently "open" across tokens
    active_spans_on_previous_token = {}

    # Read headers first to know the order of entity types
    headers = read_webanno_headers(tsv_file_path)
    if not headers:
        logging.error("Could not read headers from TSV for BIO conversion. Aborting.")
        return []

    header_idx_to_entity_type = {i: h for i, h in enumerate(headers)}

    try:
        with open(tsv_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line: # Empty line, usually separates sentences
                    if current_tokens: # End of a sentence, save it
                        bio_data.append({"tokens": current_tokens, "labels": current_labels})
                        logging.debug(f"  Saved sentence {len(bio_data)}: {current_tokens[:5]}... {current_labels[:5]}...")
                        current_tokens = []
                        current_labels = []
                        active_spans_on_previous_token = {} # Reset active spans for new sentence
                    continue
                
                if line.startswith("#"): # Header or text line
                    if line.startswith("#Text="):
                        if current_tokens: # If there are pending tokens, save them as a sentence before starting new #Text block
                            bio_data.append({"tokens": current_tokens, "labels": current_labels})
                            logging.debug(f"  Saved sentence {len(bio_data)} (from #Text= boundary): {current_tokens[:5]}... {current_labels[:5]}...")
                            current_tokens = []
                            current_labels = []
                            active_spans_on_previous_token = {} # Reset active spans for new sentence
                    continue # Skip processing header/text lines as tokens

                # Process token line
                parts = line.split('\t')
                # A WebAnno TSV 3.3 token line has at least 3 fixed columns + N annotation columns
                if len(parts) < 3 + len(headers): 
                    logging.warning(f"Line {line_num} has unexpected number of columns: {line}. Expected at least {3 + len(headers)}, got {len(parts)}. Skipping.")
                    continue

                token_text = parts[2]
                annotation_columns_raw = parts[3:] # Annotations start from the 4th column (index 3)

                # This will store the BIO tag for each *specific header column* for the current token
                # e.g., ["O", "B-Locationofstudy", "O", ...]
                token_bio_labels_per_header = ["O"] * len(headers) 
                
                # Track spans that are active *on the current token*
                active_spans_on_current_token = {}

                # Iterate through each annotation column for the current token
                for col_idx, ann_str in enumerate(annotation_columns_raw):
                    if ann_str != "_":
                        individual_anns = ann_str.split('|')
                        for ann in individual_anns:
                            match = re.match(r"\*\[(\d+)\]", ann)
                            if match:
                                span_id = int(match.group(1))
                                entity_type = header_idx_to_entity_type.get(col_idx)
                                if entity_type is None:
                                    logging.warning(f"Line {line_num}: Entity type for column {col_idx} (header index) not found in headers. Skipping annotation '{ann}'.")
                                    continue

                                # Check if this span was active on the *previous* token
                                if span_id in active_spans_on_previous_token and active_spans_on_previous_token[span_id] == entity_type:
                                    # It was active and the type matches, so it's 'I-'
                                    token_bio_labels_per_header[col_idx] = f"I-{entity_type}"
                                    logging.debug(f"    Line {line_num}, Token '{token_text}': Span {span_id} continued as I-{entity_type}")
                                else:
                                    # It's a new span or a span that ended and restarted, so it's 'B-'
                                    token_bio_labels_per_header[col_idx] = f"B-{entity_type}"
                                    logging.debug(f"    Line {line_num}, Token '{token_text}': Span {span_id} started as B-{entity_type}")
                                
                                # Mark this span as active for the *current* token
                                active_spans_on_current_token[span_id] = entity_type
                            else:
                                logging.warning(f"Line {line_num}: Malformed annotation '{ann}'. Skipping.")
                
                # Update active spans for the next iteration (next token)
                active_spans_on_previous_token = active_spans_on_current_token

                current_tokens.append(token_text)
                # For the 'labels' list in BIO data, we need a single label per token.
                # If a token is part of multiple entities (e.g., overlapping spans of different types),
                # this logic needs a rule. For now, we'll prioritize the first non-'O' label found.
                final_token_label = "O"
                for label_val in token_bio_labels_per_header:
                    if label_val != "O":
                        final_token_label = label_val
                        break
                current_labels.append(final_token_label)
                logging.debug(f"  Token '{token_text}' final BIO label: {final_token_label}")

        # After loop, if there are any remaining tokens, add the last sentence
        if current_tokens:
            bio_data.append({"tokens": current_tokens, "labels": current_labels})
            logging.debug(f"  Saved final sentence {len(bio_data)}: {current_tokens[:5]}... {current_labels[:5]}...")

    except FileNotFoundError:
        logging.error(f"Error: TSV file '{tsv_file_path}' not found.")
    except Exception as e:
        logging.error(f"An error occurred during TSV to BIO conversion: {e}", exc_info=True)
    
    logging.info(f"Finished WebAnno TSV to BIO conversion. Total sentences: {len(bio_data)}")
    return bio_data


# --- Part 1: Data Preparation for Instruction Fine-tuning ---

def transform_bio_to_instruction_format(bio_data, ner_labels_list):
    """
    Transforms a dataset from BIO format to instruction-response pairs.
    Each example will have an 'instruction', 'input_text', and 'output_json_string'.
    The output JSON will be a list of objects: [{"entity_text": "...", "entity_type": "...", "start_char": ..., "end_char": ...}]
    The instruction prompt is updated to explicitly list the allowed entity types.
    """
    logging.info("Starting BIO to Instruction format transformation.")
    instruction_data = []
    
    # Create a string of allowed entity types for the instruction
    allowed_types_str = ", ".join(ner_labels_list)

    for item_idx, item in enumerate(bio_data):
        logging.debug(f"Processing item {item_idx + 1}/{len(bio_data)}")
        tokens = item["tokens"]
        labels = item["labels"]

        # Reconstruct full text and track character offsets for each token
        full_text_parts = []
        token_char_offsets = []
        current_offset = 0
        for token in tokens:
            if full_text_parts: # Add space before token if not the first
                full_text_parts.append(" ")
                current_offset += 1
            token_start = current_offset
            full_text_parts.append(token)
            current_offset += len(token)
            token_end = current_offset
            token_char_offsets.append((token_start, token_end))
        full_text = "".join(full_text_parts)
        logging.debug(f"  Full text reconstructed: '{full_text}'")
        logging.debug(f"  Token character offsets: {token_char_offsets}")

        extracted_entities_list = [] # Will store [{"entity_text": "...", "entity_type": "...", "start_char": ..., "end_char": ...}]
        current_entity_tokens = []
        current_entity_type = None
        current_entity_start_char = -1

        for i, (token, label) in enumerate(zip(tokens, labels)):
            token_start_char, token_end_char = token_char_offsets[i]
            logging.debug(f"    Token: '{token}', Label: '{label}', Char Span: ({token_start_char}, {token_end_char})")

            if label.startswith("B-"):
                if current_entity_tokens: # Save previous entity if exists
                    extracted_entities_list.append({
                        "entity_text": " ".join(current_entity_tokens),
                        "entity_type": current_entity_type,
                        "start_char": current_entity_start_char,
                        "end_char": token_char_offsets[i-1][1] # End of previous token
                    })
                    logging.debug(f"      Saved entity: '{' '.join(current_entity_tokens)}': '{current_entity_type}' ({current_entity_start_char}-{token_char_offsets[i-1][1]})")
                
                current_entity_type = label[2:] # Remove "B-"
                current_entity_tokens = [token]
                current_entity_start_char = token_start_char
                logging.debug(f"      Started new entity '{current_entity_type}' with token '{token}' at char {current_entity_start_char}")
            elif label.startswith("I-"):
                if current_entity_type and label[2:] == current_entity_type: # Continue current entity
                    current_entity_tokens.append(token)
                    logging.debug(f"      Continued entity '{current_entity_type}' with token '{token}'")
                else: # Mismatch or I- without preceding B-, treat as new B- or O
                    if current_entity_tokens: # Save previous entity if exists
                        extracted_entities_list.append({
                            "entity_text": " ".join(current_entity_tokens),
                            "entity_type": current_entity_type,
                            "start_char": current_entity_start_char,
                            "end_char": token_char_offsets[i-1][1]
                        })
                        logging.debug(f"      Saved entity (I-mismatch): '{' '.join(current_entity_tokens)}': '{current_entity_type}' ({current_entity_start_char}-{token_char_offsets[i-1][1]})")
                    current_entity_type = label[2:] # Treat as new B-
                    current_entity_tokens = [token]
                    current_entity_start_char = token_start_char
                    logging.debug(f"      Started new entity (I-mismatch) '{current_entity_type}' with token '{token}' at char {current_entity_start_char}")
            else: # 'O' label
                if current_entity_tokens: # Save previous entity if exists
                    extracted_entities_list.append({
                        "entity_text": " ".join(current_entity_tokens),
                        "entity_type": current_entity_type,
                        "start_char": current_entity_start_char,
                        "end_char": token_char_offsets[i-1][1]
                    })
                    logging.debug(f"      Saved entity (O-label): '{' '.join(current_entity_tokens)}': '{current_entity_type}' ({current_entity_start_char}-{token_char_offsets[i-1][1]})")
                current_entity_tokens = []
                current_entity_type = None
                current_entity_start_char = -1
                logging.debug(f"      Encountered 'O' label, reset current entity.")

        # After loop, save any pending entity
        if current_entity_tokens:
            extracted_entities_list.append({
                "entity_text": " ".join(current_entity_tokens),
                "entity_type": current_entity_type,
                "start_char": current_entity_start_char,
                "end_char": token_char_offsets[len(tokens)-1][1] # End of the last token
            })
            logging.debug(f"  Saved final pending entity: '{' '.join(current_entity_tokens)}': '{current_entity_type}' ({current_entity_start_char}-{token_char_offsets[len(tokens)-1][1]})")

        # Format the output as a JSON string (list of objects)
        output_json_string = json.dumps(extracted_entities_list, ensure_ascii=False)
        # Add markdown code block for training consistency
        formatted_output = f"```json\n{output_json_string}\n```"
        logging.debug(f"  Generated JSON output for training: {formatted_output}")

        # Updated instruction to explicitly list allowed entity types
        instruction_prompt = (
            f"Extract entities from the following text and output them as a JSON array of objects, "
            f"where each object has 'entity_text', 'entity_type', 'start_char', and 'end_char' keys. "
            f"Only use the following entity types: {allowed_types_str}. "
            f"Only include the JSON array within a ```json block."
        )

        instruction_data.append({
            "instruction": instruction_prompt,
            "input": full_text,
            "output": formatted_output
        })
    logging.info("Finished BIO to Instruction format transformation.")
    return instruction_data

def prepare_dataset_for_finetuning_instruction(data, tokenizer):
    """
    Prepares the instruction-response dataset for causal language modeling fine-tuning.
    It applies the chat template and masks the prompt tokens so the model only learns to generate the response part.
    Explicitly pads input_ids, attention_mask, and labels to MAX_SEQ_LENGTH.
    """
    logging.info("Starting dataset preparation for instruction fine-tuning.")
    processed_examples = []
    for example_idx, example in enumerate(data):
        logging.debug(f"Preparing example {example_idx + 1}/{len(data)}")
        messages = [
            {"role": "user", "content": f"{example['instruction']}\n\nText: {example['input']}"},
            {"role": "assistant", "content": example['output']}
        ]
        logging.debug(f"  Messages for chat template: {messages}")

        # Apply the chat template to get the full prompt and response sequence string
        chat_string = tokenizer.apply_chat_template(
            messages,
            tokenize=False, # Get string output first
            add_generation_prompt=True
        )
        
        # Now tokenize and pad/truncate the string to MAX_SEQ_LENGTH
        tokenized_output = tokenizer(
            chat_string,
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
            padding="max_length", # Explicitly pad to max_length
            return_tensors="pt"
        )
        
        input_ids = tokenized_output["input_ids"][0]
        attention_mask = tokenized_output["attention_mask"][0]
        logging.debug(f"  Tokenized chat input_ids length: {len(input_ids)}")

        # Create labels for causal language modeling.
        # Tokenize only the assistant's content to get its token IDs
        assistant_output_ids = tokenizer.encode(example['output'], add_special_tokens=False)
        logging.debug(f"  Assistant output token IDs length: {len(assistant_output_ids)}")

        assistant_content_start_idx = -1
        # Iterate through the input_ids to find the sequence of assistant_output_ids
        for i in range(len(input_ids) - len(assistant_output_ids) + 1):
            if input_ids[i:i+len(assistant_output_ids)].tolist() == assistant_output_ids:
                assistant_content_start_idx = i
                break

        labels = torch.full_like(input_ids, -100) # Initialize labels with -100 (ignore in loss)

        if assistant_content_start_idx != -1:
            labels[assistant_content_start_idx:] = input_ids[assistant_content_start_idx:]
            logging.debug(f"  Assistant content starts at index: {assistant_content_start_idx}")
        else:
            logging.warning(f"  Could not find assistant's content in tokenized sequence for example {example_idx + 1}. Training on full sequence.")
            labels = input_ids # Fallback: if content not found, train on everything

        processed_examples.append({
            "input_ids": input_ids.tolist(),
            "attention_mask": attention_mask.tolist(),
            "labels": labels.tolist()
        })

    dataset = Dataset.from_dict({
        "input_ids": [ex["input_ids"] for ex in processed_examples],
        "attention_mask": [ex["attention_mask"] for ex in processed_examples],
        "labels": [ex["labels"] for ex in processed_examples]
    })
    logging.info(f"Finished dataset preparation. Prepared instruction dataset size: {len(dataset)}")
    return dataset


def fine_tune_model_instruction(train_dataset, tokenizer, model):
    """
    Performs the instruction fine-tuning process using Hugging Face Trainer.
    """
    logging.info("Starting fine-tuning process.")
    model.gradient_checkpointing_enable()
    logging.info("Gradient checkpointing enabled.")
    
    # Always prepare for k-bit training if it's a PEFT model, regardless of explicit CUDA check here.
    # The actual loading of bitsandbytes will handle the device.
    model = prepare_model_for_kbit_training(model)
    logging.info("Model prepared for k-bit training.")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    logging.info("PEFT LoRA model configured.")
    model.print_trainable_parameters()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    logging.info("Data collator for causal language modeling initialized.")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        per_device_eval_batch_size=1,
        do_eval=False, # Replaced evaluation_strategy="no"
        fp16=True, # Always attempt fp16 if model is loaded with bfloat16/float16
        report_to="none",
    )
    logging.info(f"Training arguments configured: {training_args}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logging.info("Trainer initialized. Starting training...")
    trainer.train()
    logging.info("Fine-tuning complete. Saving model...")

    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logging.info(f"Fine-tuned model and tokenizer saved to {OUTPUT_DIR}")


# --- Part 2: Inference with the Instruction Fine-tuned Model ---

def load_fine_tuned_model(model_path, model_name=MODEL_NAME):
    """
    Loads the base model and then the fine-tuned PEFT adapter.
    """
    logging.info(f"Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left" # Qwen is typically left-padded for generation
        logging.info(f"Tokenizer pad_token set to EOS token: {tokenizer.pad_token_id}, padding_side: {tokenizer.padding_side}")

    # Always attempt 4-bit loading if CUDA is available, as requested
    load_in_4bit = torch.cuda.is_available()
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    if load_in_4bit:
        logging.info("CUDA available. Attempting to load model in 4-bit with bfloat16/float16.")
    else:
        logging.warning("CUDA not available. Loading model without 4-bit quantization. This may require significant CPU RAM.")
        torch_dtype = torch.float32 # Fallback to float32 on CPU if no CUDA

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=load_in_4bit,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True
        )
        logging.info(f"Base model loaded on device: {model.device}")
    except Exception as e:
        logging.error(f"Error loading base model: {e}")
        if torch.cuda.is_available() and "bitsandbytes" in str(e).lower():
            logging.error("This error often means your `bitsandbytes` library was not compiled with CUDA support.")
            logging.error("Please ensure you have installed `bitsandbytes` correctly for your CUDA version (e.g., `pip install bitsandbytes-cuda11x` or `pip install bitsandbytes-cuda12x`).")
            logging.error("You can also try running `python -c 'import torch; print(torch.cuda.is_available()); import bitsandbytes as bnb; print(bnb.cuda.is_available())'` to check your setup.")
        raise # Re-raise the exception after providing troubleshooting tips

    try:
        from peft import PeftModel
        logging.info(f"Attempting to load PEFT adapter from {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload() # Merge LoRA weights back into the base model for easier inference
        logging.info("PEFT adapter loaded and merged successfully.")
    except Exception as e:
        logging.warning(f"Could not load PEFT adapter from {model_path}. Proceeding with base model. Error: {e}")
        logging.warning("This might happen if you haven't fine-tuned yet or the path is incorrect.")

    model.eval() # Set model to evaluation mode
    logging.info("Model set to evaluation mode.")
    return tokenizer, model

def predict_entities_from_text_instruction(text_content, tokenizer, model, ner_labels_list):
    """
    Processes a text string to identify entities using the instruction-tuned model
    and returns them in the specified JSON format.
    The instruction prompt explicitly lists the allowed entity types.
    """
    logging.info("Starting entity prediction using instruction-tuned model.")
    
    allowed_types_str = ", ".join(ner_labels_list)
    instruction = (
        f"Extract entities from the following text and output them as a JSON array of objects, "
        f"where each object has 'entity_text', 'entity_type', 'start_char', and 'end_char' keys. "
        f"Only use the following entity types: {allowed_types_str}. "
        f"Only include the JSON array within a ```json block."
    )
    
    messages = [
        {"role": "user", "content": f"{instruction}\n\nText: {text_content}"}
    ]
    logging.debug(f"Constructed messages for generation: {messages}")
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    logging.debug(f"Input IDs for generation (length {len(input_ids[0])}): {input_ids[0].tolist()}")

    logging.info("Generating response from model...")
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=GENERATION_MAX_NEW_TOKENS,
            do_sample=False, # For deterministic output
            pad_token_id=tokenizer.eos_token_id, # Important for padding
            eos_token_id=tokenizer.eos_token_id # Stop generation at EOS
        )
    logging.info("Response generation complete.")

    # Decode the generated text, skipping the input prompt part
    generated_text = tokenizer.decode(generated_ids[0][len(input_ids[0]):], skip_special_tokens=True)
    logging.info(f"Raw generated text: {generated_text}")

    # Attempt to parse the JSON from the generated text
    # Look for the ```json block and extract the content within it
    json_match = re.search(r"```json\s*(\[.*?\])\s*```", generated_text, re.DOTALL)
    
    # This will be the final list of (entity_text, entity_type, start_char, end_char) tuples
    extracted_entities_for_webanno = [] 

    if json_match:
        json_array_string = json_match.group(1) # Capture group 1 is the content inside the []
        logging.debug(f"Found potential JSON array string: {json_array_string}")
        try:
            extracted_json_list = json.loads(json_array_string)
            logging.info("Successfully parsed JSON array from model output.")
            
            # Convert list of {"entity_text": "...", "entity_type": "..."} to tuples
            for item in extracted_json_list:
                if "entity_text" in item and "entity_type" in item and "start_char" in item and "end_char" in item:
                    # Validate entity type against NER_LABELS
                    if item["entity_type"] in ner_labels_list:
                        extracted_entities_for_webanno.append((
                            item["entity_text"],
                            item["entity_type"],
                            item["start_char"],
                            item["end_char"]
                        ))
                    else:
                        logging.warning(f"Skipping entity with unrecognized type: {item['entity_type']} for text '{item['entity_text']}'.")
                else:
                    logging.warning(f"Skipping malformed entity item (missing keys): {item}")

        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON array from model output: {e}")
            logging.error(f"Attempted JSON string: {json_array_string}")
    else:
        logging.warning("No ```json array block found in the model's generated output.")

    # Assign a unique ID to each extracted entity for WebAnno, starting from 1
    # This ensures each span in the output TSV has a unique ID, as required by WebAnno
    entities_with_ids = []
    current_span_id = 1
    for entity_text, entity_type, start_char, end_char in extracted_entities_for_webanno:
        entities_with_ids.append((current_span_id, entity_text, entity_type, start_char, end_char))
        current_span_id += 1

    logging.info(f"Extracted entities for WebAnno conversion (with IDs): {entities_with_ids}")
    return entities_with_ids

# --- Part 3: Converting JSON Output to WebAnno 3.3 .tsv ---

def find_all_spans(text, substring):
    """Find all (start, end) spans of substring in text (non-overlapping)."""
    results = []
    idx = text.find(substring)
    while idx != -1:
        results.append((idx, idx + len(substring)))
        idx = text.find(substring, idx + 1)
    return results

def map_model_entities_to_text_spans(model_entities, text, headers):
    """For each entity in the model output, find its true start/end char positions in the input text."""
    used_spans = set()
    span_tuples = []
    for entity in model_entities:
        # Accept both dict or tuple
        if isinstance(entity, dict):
            entity_text = entity["entity_text"]
            entity_type = entity["entity_type"]
        elif isinstance(entity, (tuple, list)) and len(entity) >= 2:
            entity_text = entity[0]
            entity_type = entity[1]
        else:
            print(f"[WARN] Unexpected entity format: {entity}")
            continue
        if entity_type not in headers:
            print(f"[WARN] Skipping entity with unknown type: {entity_type}")
            continue
        matches = find_all_spans(text, entity_text)
        if not matches:
            print(f"[WARN] Entity text not found in input: {repr(entity_text)}")
        for start, end in matches:
            if (start, end, entity_type) not in used_spans:
                span_tuples.append((start, end, entity_type))
                used_spans.add((start, end, entity_type))
                break  # Only first match per entity_text
    return span_tuples


def json_to_webanno(spans, text, headers):
    """
    Convert character-offset spans to WebAnno TSV 3.3 lines.
    Each span is (start_char, end_char, entity_type).
    Headers are the annotation columns in order.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    sentences = list(doc.sents)
    lines = ["#FORMAT=WebAnno TSV 3.3"]
    for h in headers:
        lines.append(f"#T_SP=webanno.custom.{h}|")

    annotation_id_counter = 1
    active_span_annotations = {}

    for sent_idx, sent in enumerate(sentences, start=1):
        lines.append(f"\n#Text={sent.text}")
        for tok_idx, tok in enumerate(sent, start=1):
            token_text = tok.text
            token_global_start = tok.idx
            token_global_end = tok.idx + len(token_text)
            labels_for_token = {}
            for s_start, s_end, s_label in spans:
                # Overlap: token and span
                if max(token_global_start, s_start) < min(token_global_end, s_end):
                    span_key = (s_start, s_end, s_label)
                    if span_key not in active_span_annotations:
                        active_span_annotations[span_key] = annotation_id_counter
                        annotation_id_counter += 1
                    ann_id = active_span_annotations[span_key]
                    col_idx = headers.index(s_label)
                    labels_for_token.setdefault(col_idx, []).append(ann_id)
            # Write global char offsets (not per sentence)
            row = [f"{sent_idx}-{tok_idx}", f"{token_global_start}-{token_global_end}", token_text]
            for h_idx in range(len(headers)):
                if h_idx in labels_for_token:
                    row.append(";".join(f"*[{aid}]" for aid in sorted(labels_for_token[h_idx])))
                else:
                    row.append("_")
            lines.append("\t".join(row))
    return lines


# --- Example Usage ---

if __name__ == "__main__":
    logging.info("Script started.")

    # --- Define File Paths and Output Directory ---
    input_text_file_path = "data/fine-tuned/txt/(3) Bulmer2024.txt" # Create this file with text you want to annotate
    human_annotated_data_path = "data/fine-tuned/tsv/(3) Bulmer2024.tsv" # User's provided human-annotated file for fine-tuning data
    input_tsv_for_headers_path = "data/fine-tuned/tsv/(3) Bulmer2024.tsv" # Create this file with your desired #T_SP headers for output
    output_directory = "./output-finetuned"
    
    # Dynamically set output_tsv_filename based on input_text_file_path
    base_filename = os.path.splitext(os.path.basename(input_text_file_path))[0]
    output_tsv_filename = f"{base_filename}.tsv"
    output_json_filename = f"{base_filename}.json" # JSON output filename

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)
    logging.info(f"Output directory ensured: {output_directory}")

    # --- Step 0: Read NER Labels from TSV Header (for output TSV) ---
    NER_LABELS_FOR_OUTPUT = read_webanno_headers(input_tsv_for_headers_path)
    if not NER_LABELS_FOR_OUTPUT:
        logging.error("No NER labels could be loaded from the template TSV for output. Exiting.")
        exit() # Exit if we can't get any labels

    # --- Check if model is already fine-tuned ---
    model_already_finetuned = False
    if os.path.exists(OUTPUT_DIR) and (os.path.exists(os.path.join(OUTPUT_DIR, "adapter_model.safetensors")) or os.path.exists(os.path.join(OUTPUT_DIR, "pytorch_model.bin"))):
        logging.info(f"Fine-tuned model found in '{OUTPUT_DIR}'. Skipping fine-tuning.")
        model_already_finetuned = True
    else:
        logging.info(f"Fine-tuned model not found in '{OUTPUT_DIR}'. Proceeding with fine-tuning.")


    # --- Step 1: Data Transformation and Fine-tuning (Conditional) ---
    if not model_already_finetuned:
        logging.info("--- Starting Data Transformation and Fine-tuning Section ---")
        raw_bio_data = read_webanno_tsv_to_bio(human_annotated_data_path)

        # Initialize tokenizer before fine-tuning or dataset preparation
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            logging.info(f"Tokenizer pad_token set to EOS token: {tokenizer.pad_token_id}, padding_side: {tokenizer.padding_side}")

        if raw_bio_data:
            logging.info(f"Loaded {len(raw_bio_data)} samples from '{human_annotated_data_path}' for fine-tuning.")
            instruction_finetune_data = transform_bio_to_instruction_format(raw_bio_data, NER_LABELS_FOR_OUTPUT)
            logging.info(f"Transformed {len(instruction_finetune_data)} samples for instruction fine-tuning.")
            logging.info("Example transformed data point (first 500 chars):")
            logging.info(json.dumps(instruction_finetune_data[0], indent=2)[:500] + "...")

            train_dataset = prepare_dataset_for_finetuning_instruction(instruction_finetune_data, tokenizer)
            logging.info(f"Prepared instruction dataset size: {len(train_dataset)}")

            logging.info("Initializing model for instruction fine-tuning.")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                load_in_4bit=torch.cuda.is_available(),
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            logging.info(f"Model loaded in 4-bit on device: {model.device}")

            fine_tune_model_instruction(train_dataset, tokenizer, model)
            logging.info("--- Instruction fine-tuning complete. ---")
        else:
            logging.warning(f"No data loaded for fine-tuning from '{human_annotated_data_path}'. Skipping fine-tuning step.")
        logging.info("--- Finished Data Transformation and Fine-tuning Section ---")
    else:
        # If fine-tuning is skipped, we still need to initialize the tokenizer for inference
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            logging.info(f"Tokenizer pad_token set to EOS token: {tokenizer.pad_token_id}, padding_side: {tokenizer.padding_side}")


    # --- Step 2: Inference ---
    logging.info("--- Starting Inference Section ---")
    inference_tokenizer, inference_model = load_fine_tuned_model(OUTPUT_DIR, MODEL_NAME)
    logging.info("Model and tokenizer loaded for inference.")

    # Read text from the input .txt file with robust encoding handling
    text_to_annotate = ""
    try:
        with open(input_text_file_path, 'r', encoding='utf-8') as f:
            text_to_annotate = f.read()
        logging.info(f"Loaded text from: {input_text_file_path} with UTF-8 encoding.")
    except UnicodeDecodeError:
        logging.warning(f"UnicodeDecodeError with UTF-8 for '{input_text_file_path}'. Trying latin-1 encoding.")
        try:
            with open(input_text_file_path, 'r', encoding='latin-1') as f:
                text_to_annotate = f.read()
            logging.info(f"Loaded text from: {input_text_file_path} with latin-1 encoding.")
        except Exception as e:
            logging.error(f"Error reading input text file '{input_text_file_path}' with latin-1: {e}. Using dummy text.")
            text_to_annotate = """
            Blue carbon habitats in Aotearoa New Zealand—opportunities for conservation, restoration, and carbon sequestration
            Richard H Bulmer, Phoebe J Stewart_Sinclair, Orlando Lam_Gordillo, Stephanie Mangan, Luitgard Schwendenmann, Carolyn J Lundquist
            First published: 30 July 2024
            https://doi.org/10.1111/rec.14225
            Author contributions: all authors contributed to the conceptual development of the work; RHB, PJS_S, OL_G, SM conducted the analysis; RHB, PJS_S led the writing; OL_G, SM, LS, CJL reviewed the manuscript; RHB, CJL secured funding.
            """
    except FileNotFoundError:
        logging.error(f"Error: Input text file '{input_text_file_path}' not found. Using dummy text.")
        text_to_annotate = """
        Blue carbon habitats in Aotearoa New Zealand—opportunities for conservation, restoration, and carbon sequestration
        Richard H Bulmer, Phoebe J Stewart_Sinclair, Orlando Lam_Gordillo, Stephanie Mangan, Luitgard Schwendenmann, Carolyn J Lundquist
        First published: 30 July 2024
        https://doi.org/10.1111/rec.14225
        Author contributions: all authors contributed to the conceptual development of the work; RHB, PJS_S, OL_G, SM conducted the analysis; RHB, PJS_S led the writing; OL_G, SM, LS, CJL reviewed the manuscript; RHB, CJL secured funding.
        """
    except Exception as e:
        logging.error(f"An unexpected error occurred while reading '{input_text_file_path}': {e}. Using dummy text.")
        text_to_annotate = """
        Blue carbon habitats in Aotearoa New Zealand—opportunities for conservation, restoration, and carbon sequestration
        Richard H Bulmer, Phoebe J Stewart_Sinclair, Orlando Lam_Gordillo, Stephanie Mangan, Luitgard Schwendenmann, Carolyn J Lundquist
        First published: 30 July 2024
        https://doi.org/10.1111/rec.14225
        Author contributions: all authors contributed to the conceptual development of the work; RHB, PJS_S, OL_G, SM conducted the analysis; RHB, PJS_S led the writing; OL_G, SM, LS, CJL reviewed the manuscript; RHB, CJL secured funding.
        """

    # --- Normalize text_to_annotate BEFORE passing to the model for inference ---
    # This ensures the model's output offsets are consistent with the text spaCy will process.
    normalized_text_to_annotate = text_to_annotate.replace('\r\n', '\n').replace('\r', '\n')
    logging.info("Input text normalized for consistent newline handling.")

    logging.info(f"\nInput Text for Inference (first 200 chars after normalization):\n{normalized_text_to_annotate[:200]}...") # Log first 200 chars

    logging.info("Extracting entities using instruction-tuned model.")
    # Pass the normalized text to the prediction function
    extracted_entities_for_webanno = predict_entities_from_text_instruction(normalized_text_to_annotate, inference_tokenizer, inference_model, NER_LABELS_FOR_OUTPUT)
    
    # Convert the list of tuples to the desired dictionary format for display in logs
    extracted_json_for_display = {item[0]: item[1] for item in extracted_entities_for_webanno}
    logging.info("\nExtracted Entities (JSON Output for Display):")
    logging.info(json.dumps(extracted_json_for_display, indent=4))
    logging.info("--- Finished Inference Section ---")

    # --- Step 3: WebAnno TSV Conversion ---
        # --- Step 3: WebAnno TSV Conversion (using robust mapping) ---
    logging.info("--- Starting WebAnno TSV Conversion Section ---")
    logging.info("Mapping model entities to true character offsets in text...")
    spans_for_webanno_func = map_model_entities_to_text_spans(
        extracted_entities_for_webanno,
        normalized_text_to_annotate,
        NER_LABELS_FOR_OUTPUT
    )
    logging.info(f"Mapped {len(spans_for_webanno_func)} entities.")

    logging.info("Converting mapped spans to WebAnno TSV...")
    webanno_tsv_output_lines = json_to_webanno(spans_for_webanno_func, normalized_text_to_annotate, NER_LABELS_FOR_OUTPUT)
    webanno_tsv_output = "\n".join(webanno_tsv_output_lines)

    # Save the output to a .tsv file in the specified directory
    output_tsv_path = os.path.join(output_directory, output_tsv_filename)
    with open(output_tsv_path, "w", encoding="utf-8") as f:
        f.write(webanno_tsv_output)
    logging.info(f"WebAnno TSV saved to {output_tsv_path}")
    logging.info("--- Finished WebAnno TSV Conversion Section ---")
    logging.info("Script finished.")

