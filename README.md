Here's the updated README with the link to the pre-trained corpus added:

---

# SansGPT: Advancing Generative Pre-Training in Sanskrit

This repository contains the codebase for **SansGPT**, a generative pre-trained model for Sanskrit. The project includes tools for pre-training and fine-tuning the model on tasks like Semantic Analogy Prediction and Simile Element Extraction.

---

## Usage

- **To pre-train the model**, run `Pre-Train-Pipeline.py`.  
- **To fine-tune the model** on specific NLP tasks, run `Fine-Tune-Pipeline.py`.

---

## Pre-Trained Corpus
You can download the pre-trained Sanskrit text corpus [here](https://drive.google.com/file/d/18SrpFJCbrDyR5420RYMoHc0n37a_6D4h/view?usp=sharing).

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

--- 

This addition makes the README more user-friendly by providing direct access to the pre-trained corpus.
