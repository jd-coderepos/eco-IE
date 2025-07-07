import json
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)

# === Configuration ===
MODEL_NAME   = "Qwen/Qwen3-4B"
TRAIN_FILE   = "output_new/ner_train.json"
VALID_FILE   = "output_new/ner_val.json"
TEST_FILE    = "output_new/ner_test.json"
OUTPUT_DIR   = "fine_tuned_qwen3_ner"
MAX_LENGTH   = 512  # reduce for efficiency and stability
BATCH_SIZE   = 2    # increase if memory allows
LEARNING_RATE= 2e-5
NUM_EPOCHS   = 3
GRAD_ACCUM_STEPS = 4

# === Instruction Prompt ===
INSTRUCTION = """
You are an expert in Named Entity Recognition. Given the following text, output the BIO tags for each token as a space-separated sequence.
"""

# --- Load tokenizer & model ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
model.config.use_cache = False

# === Preprocessing Function ===
def preprocess_function(example):
    tokens = example["tokens"]
    labels = example["labels"]
    text = " ".join(tokens)
    tags_str = " ".join(labels)

    # prompt + target
    prompt = f"{INSTRUCTION}\nText: {text}\nTags:"
    full_input = prompt + " " + tags_str + tokenizer.eos_token
    dotted_line = "-" * 80
    print(dotted_line)
    print("full_input:", full_input)
    print(dotted_line)

    # tokenize with fixed max length and padding
    tokenized = tokenizer(
        full_input,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    # mask prompt tokens
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False
    )["input_ids"]
    prompt_len = len(prompt_ids)

    # Ensure prompt_len does not exceed input length
    prompt_len = min(prompt_len, len(input_ids))

    labels_ids = input_ids.copy()
    # mask prompt portion and pad
    for i in range(prompt_len):
        labels_ids[i] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels_ids
    }

# === Load dataset ===
data_files = {"train": TRAIN_FILE, "validation": VALID_FILE, "test": TEST_FILE}
datasets = load_dataset("json", data_files=data_files)

# === Preprocess & tokenize ===
tokenized_datasets = datasets.map(
    preprocess_function,
    remove_columns=datasets["train"].column_names,
    batched=False
)

# set format for Trainer
for split in ["train", "validation"]:
    tokenized_datasets[split].set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

# === Data collator for token classification ===
data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    label_pad_token_id=-100,
    padding=True
)

# === Training arguments ===
use_cuda = torch.cuda.is_available()
# Only one of bf16/fp16
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    fp16=False,
    bf16=use_cuda,
    gradient_checkpointing=True,
    max_grad_norm=1.0,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    load_best_model_at_end=True,
    save_total_limit=2,
)

# === Initialize Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# === Train ===
trainer.train()
trainer.save_model(OUTPUT_DIR)

# === Inference Function ===
def infer_ner(text: str, max_new_tokens: int = 256) -> list:
    model.eval()
    prompt = f"{INSTRUCTION}\nText: {text}\nTags:"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True
    ).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    gen = outputs[0][inputs.input_ids.shape[-1]:]
    tag_seq = tokenizer.decode(gen, skip_special_tokens=True).strip()
    return tag_seq.split()

# === Example usage ===
if __name__ == "__main__":
    # Example inference on a text file
    sample_text = """Author contributions: EPA, UI, DA, KCG conceived and designed the research; EPA, DA, KCG implemented the experiment and collected experimental data; EPA analyzed the data; EPA, KCG led the writing of the manuscript; all authors contributed significantly in writing, reviewing, and editing.
Coordinating Editor: Stephen Murphy.
Abstract.
While much research has focused on genetic variation in plants in relation to abiotic clines in temperate and boreal forests, few studies have examined similar relationships in tropical forests. Genetic variation in desirable performance traits of trees, such as drought tolerance, fast_growth, and carbon sequestration rates, is widely used to improve reforestation efforts in nontropical systems. However, evolutionary processes such as local adaptation are poorly understood in tropical forests making it difficult to locate desired phenotypes. To test for genetic variation in growth rate in relationship to climatic clines, we conducted a common garden study over 18 months in a nursery using four dipterocarp tree species, represented by 9_12 half_sib families, sourced across an elevational gradient ranging from lowland to hill forests (circa 130_470 m above sea_level) in Malaysian Borneo. We found genetic variation in growth for all four species with fast_growing half_sib families growing 42_88% faster than poorly performing half_sib families. Furthermore, in three species we found that elevation of seedling origin predicted seedling performance; in Shorea fallax and S. johorensis, half_sib families originating from low elevations performed the best. In S. argentifolia half_sib families' seedlings from low elevations grew slowly. Because elevation is a good proxy for climate, the finding of elevational clines predicting genetic variation in growth provides evidence of evolution affecting the function of tropical tree species. Our research highlights opportunities to better understand evolutionary processes in tropical forests and to use such information to improve seed source selection in reforestation.
Implications for Practice
* Tree species from the dipterocarp family contain considerable genetic variation in growth that is useful for enhancing restoration or plantation forestry using native tree species.
* Part of this variation in growth rate is predicted by a commonly used proxy of climate variation, that is elevation, a novel result that provides rarely found evidence of local adaptation in tropical forests.
* Increased understanding of evolutionary processes resulting in genetic variation of desirable plant traits could improve seed sourcing guidelines for use in reforestation of tropical forests.
Introduction."""
    predicted_tags = infer_ner(sample_text)
    tokens = sample_text.split()
    # Pair tokens with predicted tags
    for tok, tag in zip(tokens, predicted_tags):
        print(f"{tok}\t{tag}")
