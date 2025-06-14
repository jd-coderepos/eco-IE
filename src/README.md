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

## Project Structure
- `data/test`: Contains input and sample datasets.
- `README.md`: Project documentation.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the [APACHE LICENSE](../LICENSE).
