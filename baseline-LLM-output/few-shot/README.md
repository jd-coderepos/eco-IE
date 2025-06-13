# Few-Shot Prompt Variants for Scientific NER

This repository provides multiple few-shot prompt designs for performing **Named Entity Recognition (NER)** in scientific texts using large language models. The goal is to evaluate and compare how prompt structure and example variety affect the quality of model-generated entity annotations across the following nine predefined entity categories:

- Timeperiodofstudy  
- Locationofstudy  
- Ecosystem  
- Focalpoint  
- Method  
- Researchquestions  
- Mainhypothesisandcorrespondingresults  
- Causalstatements  
- Reccomendationsandsuggestions  

## Prompt Variants

### 🧪 Few-Shot Variant 1
**Structure:**  
- 2 single-sentence examples  
- 2 three-sentence examples  

**Purpose:**  
This mixed prompt format introduces the model to both minimal and moderately contextualized examples. The single-sentence examples help the model learn entity boundaries clearly, while the longer ones provide richer context and demonstrate how to extract entities across multiple clauses.

---

### 📘 Few-Shot Variant 2  
**Structure:**  
- 4 three-sentence examples  

**Purpose:**  
This variant emphasizes extended context in all examples. It is designed to help the model generalize better to paragraph-level text by consistently exposing it to multi-sentence reasoning and co-reference.

---

### ✏️ Few-Shot Variant 3  
**Structure:**  
- 4 single-sentence examples  

**Purpose:**  
This lightweight prompt format focuses on brevity and precision. It is well-suited for constrained inference settings and tasks where concise, atomic examples are sufficient to guide the model’s learning.

---

## Usage Notes

Each variant is formatted to guide the model toward returning a strict JSON mapping of `{ "Entity": "Label" }`. The examples follow a consistent structure and cover a variety of entity combinations to help the model learn the relationships between phrases and their semantic roles in scientific literature.


