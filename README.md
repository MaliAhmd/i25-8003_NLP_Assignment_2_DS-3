# DS5007: Natural Language Processing — Assignment 2
## Neural NLP Pipeline (Continuation of BBC Urdu NLP Pipeline)
**FAST National University of Computer and Emerging Sciences, Islamabad**

- **Student Roll Number**: `i25-8003`
- **Repository Name**: `i25-8003-NLP-Assignment2`

---

## Repository Structure

The folder structure strictly mirrors the assignment submission layout:

```
i25-8003-NLP-Assignment2/
│── i25_8003_Assignment_2_DS-3.ipynb    # Complete executed Jupyter Notebook
│── README.md                   # Setup and execution instructions
│── report.pdf                  # 2–3 page project report (Times New Roman, 12pt)
│── embeddings/
│   ├── tfidf_matrix.npy        # Term-document TF-IDF matrix (10,000 x 250)
│   ├── ppmi_matrix.npy         # PPMI co-occurrence matrix (10,000 x 10,000)
│   ├── embeddings_w2v.npy      # Skip-gram Word2Vec averaged embeddings (10,000 x 100)
│   └── word2idx.json           # Vocabulary mapping dictionary (10,000 tokens)
│── models/
│   ├── bilstm_pos.pt           # 2-layer BiLSTM model weights for POS tagging
│   ├── bilstm_ner.pt           # 2-layer BiLSTM + Linear-Chain CRF model weights for NER
│   └── transformer_cls.pt      # 4-block Pre-LN Transformer model weights for Topic Classification
└── data/
    ├── pos_train.conll         # POS training set in CoNLL format (350 sentences)
    ├── pos_val.conll           # POS validation set in CoNLL format (75 sentences)
    ├── pos_test.conll          # POS test set in CoNLL format (75 sentences)
    ├── ner_train.conll         # NER training set in CoNLL BIO format (350 sentences)
    ├── ner_val.conll           # NER validation set in CoNLL BIO format (75 sentences)
    └── ner_test.conll          # NER test set in CoNLL BIO format (75 sentences)
```

---

## Environment Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/i25-8003-NLP-Assignment2.git
   cd i25-8003-NLP-Assignment2
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Windows PowerShell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install required dependencies**:
   ```bash
   pip install torch numpy scipy matplotlib seaborn scikit-learn tqdm ipykernel
   ```

---

## Instructions to Reproduce Each Part

All components are fully implemented from scratch in PyTorch (without using banned libraries such as `gensim`, `transformers`, `nn.Transformer`, `nn.MultiheadAttention`, or `nn.TransformerEncoder`).

### Part 1: Word Embeddings
1. Run Cells 1–4 to build the 10,000-word vocabulary, construct the Term-Document matrix, and calculate TF-IDF weights saved to `embeddings/tfidf_matrix.npy`.
2. Run Cells 5–7 to compute the symmetric co-occurrence matrix ($k=5$), compute the PPMI matrix saved to `embeddings/ppmi_matrix.npy`, and display the 2-D t-SNE scatter plot.
3. Run Cells 8–12 to train the from-scratch PyTorch Skip-gram model with Negative Sampling ($d=100, K=10$), save embeddings to `embeddings/embeddings_w2v.npy`, evaluate query nearest neighbours and analogies, and compute MRR across conditions C1–C4.

### Part 2: Sequence Labeling (POS Tagging & NER)
1. Run Cells 13–17 to extract 500 stratified sentences, apply rule-based POS tagging and seed gazetteer NER annotation, and export the datasets to `data/` in CoNLL format.
2. Run Cells 18–22 to train the 2-layer BiLSTM model (fine-tuned vs. frozen) for POS tagging and the BiLSTM with scratch Linear-Chain CRF for NER with Viterbi decoding. Checkpoints are saved to `models/bilstm_pos.pt` and `models/bilstm_ner.pt`.
3. Run Cells 23–25 to evaluate POS accuracy (98.00%), plot the $12 \times 12$ confusion matrix, compute `conlleval` entity-level scores, and inspect the ablation study (A1–A4).

### Part 3: Transformer Encoder for Topic Classification
1. Run Cell 26 to categorize the 250 articles into 5 topic categories and create the stratified 70/15/15 split.
2. Run Cells 27–28 to define and train the scratch Pre-LN Transformer Encoder (4 heads, 4 blocks, sinusoidal PE buffer, learned `[CLS]` token, AdamW + cosine schedule) over 20 epochs. Checkpoint is saved to `models/transformer_cls.pt`.
3. Run Cells 29–31 to evaluate test accuracy (55.26%), plot the confusion matrix, render attention heatmaps across heads, and compare against the BiLSTM baseline.

