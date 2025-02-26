# SansGPT: Advancing Generative Pre-Training in Sanskrit

This repository contains the codebase for **SansGPT**, a generative pre-trained model for Sanskrit. The project includes tools for pre-training and fine-tuning the model on tasks like Semantic Analogy Prediction and Simile Element Extraction.

**To read the paper:** [Click here](https://aclanthology.org/2024.icon-1.50/)

---

## Usage

- **To pre-train the model**, run `Pre-Train-Pipeline.py`.  
- **To fine-tune the model** on specific NLP tasks, run `Fine-Tune-Pipeline.py`.

---

## Pre-Trained Corpus
You can download the aggregated and cleaned Sanskrit text corpus for pre-training [here](https://drive.google.com/file/d/18SrpFJCbrDyR5420RYMoHc0n37a_6D4h/view?usp=sharing).

---

## Simile Element Extraction Dataset
You can access the Simile Element Extraction Dataset [here](https://docs.google.com/spreadsheets/d/1QIR2CKXUyk5ZQ8I1vMx6d-6HJEqetqfB/edit?usp=sharing&ouid=103564942194185380640&rtpof=true&sd=true).

---

## Pre-Trained and Fine-tuned Models
You can access the pre-trained and fine-tuned models of the SansGPT series at the "SansGPT v1.0.0 Release" under the Releases Section.

---

## File Descriptions

### Tokenization and Data Processing
- **BPETokenizer.py**: Implements a custom tokenizer optimized for Sanskrit text, supporting effective tokenization of compound words.  
- **DataCleaning.py**: Handles cleaning and preprocessing of Sanskrit text data to prepare it for training.  
- **DataLoader.py**: Manages loading, shuffling, and batching of data for training and validation.

### Pre-Training
- **Pre-Train-Pipeline.py**: Orchestrates the pre-training process, including data preparation and training.  
- **TrainConfig.py**: Configures hyperparameters and settings for pre-training.  
- **TrainVocab.py**: Prepares and builds the vocabulary for the Sanskrit text corpus.  
- **Trainer.py**: Manages the training loop and optimizer configurations for pre-training.  
- **Transformer.py**: Implements the Transformer architecture used in SansGPT.

### Fine-Tuning
- **Fine-Tune-Pipeline.py**: Orchestrates the fine-tuning process for downstream NLP tasks.  
- **Fine TuneConfig.py**: Configures hyperparameters and settings for fine-tuning.  
- **FineTuner.py**: Manages the fine-tuning loop and integrates task-specific configurations.  
- **Fine_Tune_Generation.py**: Generates predictions and evaluates the model on fine-tuned tasks.  
- **FT_save_checkpoint_info.py**: Extracts and saves metrics like accuracy and loss during fine-tuning.

### Checkpoint and Utility
- **save_checkpoint_info.py**: Extracts training metrics and saves them for analysis.  
