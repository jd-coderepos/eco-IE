# Named Entity Recognition Project

This repository contains the implementation of a Named Entity Recognition (NER) system, leveraging advanced zero-shot and reasoning models for entity extraction and classification. The project is designed to handle complex scenarios where labeled data is scarce, using state-of-the-art models for zero-shot learning and reasoning.

## Models Used

### Zero-Shot Model
- **Model Name**: Qwen-2.5 14B Instruct
- **Purpose**: Performs zero-shot NER tasks, enabling entity recognition without requiring task-specific labeled data.
- **Features**: 
    - Pre-trained on diverse datasets.
    - Optimized for instruction-based tasks.
    - High accuracy in zero-shot scenarios.

### Reasoning Model
- **Model Name**: Qwen-3 14B
- **Purpose**: Enhances entity recognition by incorporating reasoning capabilities, improving performance in complex contexts.
- **Features**:
    - Advanced reasoning for ambiguous or multi-step tasks.
    - Robust handling of intricate relationships between entities.

### Few-Shot Prompting Model
- **Model Name**: Qwen-2.5 14B Instruct
- **Purpose**: Performs few-shot NER tasks by leveraging a small number of labeled examples to improve entity recognition accuracy.
- **Features**:
    - Combines pre-trained knowledge with task-specific examples.
    - Flexible and adaptable to various domains.
    - Enhanced performance compared to zero-shot scenarios.

### Fine-Tuning Model
- **Model Name**: Qwen-2.5 14B Instruct
- **Purpose**: Fine-tuned using Parameter-Efficient Fine-Tuning (PEFT) to adapt the model for specific NER tasks.
- **Features**:
    - Retains pre-trained knowledge while adapting to new datasets.
- **Methodology**:
    - Utilized PEFT techniques to fine-tune the model on labeled dataset.
    - Focused on optimizing task-specific performance without overfitting.
- **Use Case**:
    - Ideal for scenarios requiring enhanced accuracy in specialized domains.

## Features
- Zero-shot entity recognition using Qwen-2.5 14B Instruct.
- Reasoning-based entity classification using Qwen-3 14B.
- Scalable and adaptable to various domains.
- Minimal dependency on labeled datasets.

## Installation

1. Clone the repository:
     ```bash
     git clone https://github.com/jd-coderepos/eco-IE.git
     cd src
     ```

2. Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```

## Usage

### Zero-Shot NER
Run the zero-shot entity recognition model:
```bash
python zero_shot_ner.py 
```

### Reasoning-Based NER
Run the reasoning model for entity classification:
```bash
python reasoning_ner.py 
```

### Few-Shot Variants 1,2,3 NER
Run the few-shot model for entity classification:
```bash
python few_shot_new_var_1.py
```
```bash
python few_shot_new_var_2.py
```
```bash
python few_shot_new_var_3.py
```
### Fine-Tuning NER
Before running the fine-tuning script, ensure that the dataset is created using the TSV files located in `src/data/train`. Use the following command to preprocess the data:

```bash
python preprocess_webanno_split.py
```
Run the fine-tuning script to adapt the model for specific NER tasks:
```bash
python fine_tuning_qwen_v2.py
```

### Inference on Fine-Tuned Model
Run the inference script to perform NER using the fine-tuned model:
```bash
python run_inference_ner.py
```

## Project Structure
- `data/test`: Contains input and sample datasets.
- `README.md`: Project documentation.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the [APACHE LICENSE](../LICENSE).
