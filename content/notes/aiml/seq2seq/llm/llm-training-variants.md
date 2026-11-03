---
title: "Llm training variants"
date: "2026-10-28"
---

These pre-training objectives teach language models syntax, semantics, and context by predicting corrupted, omitted, or shifted parts of text.

### Autoregressive & Prefix Models

- **Causal LM (Causal Language Modeling):** Predicts the next token strictly from left to right using directional masking (standard GPT architecture). Each token only attends to preceding tokens.
- **Prefix LM:** Allows full bidirectional attention over an initial prompt ("prefix"), then switches to standard autoregressive causal generation to predict the continuation (e.g., T5 decoder/prefix mode).

### Masking & Denoising Objectives

- **Masked LM (MLM):** Masks random tokens with a special tag (e.g., `[MASK]`) and trains the model to predict the original hidden words using bidirectional context (BERT-style).
- **BERT-style MLM:** An implementation detail of MLM where masked positions are replaced 80% of the time with `[MASK]`, 10% with a random token, and 10% kept unchanged, forcing the model to evaluate all token representations.
- **MASS-style (Masked Sequence to Sequence):** Masks a contiguous span of tokens in the encoder, and uses the decoder to autoregressively reconstruct only the masked segment.
- **Span Masking:** Instead of masking individual words at random, masks contiguous sequences of tokens (spans) to make recovery harder and force deeper context comprehension (e.g., SpanBERT).
- **Sentinel Masking (Span Corruption):** Replaces entire masked spans with unique sentinel tokens (e.g., `<extra_id_0>`, `<extra_id_1>`), training the decoder to output only the sentinels followed by their missing spans (used by T5).

### Sequence & Permutation Objectives

- **Permuted LM (PLM):** Samples random factorization orders for predicting tokens while keeping standard positional embeddings, capturing bidirectional context without introducing artificial `[MASK]` tokens (used by XLNet).
- **Token Classification:** Predicts a label (such as Part-of-Speech or Named Entity) for every single token representation simultaneously.
- **Token Deletion:** Deletes random tokens entirely without leaving mask tokens; the model must reconstruct the full missing sequence (BART denoising).
- **Token Reordering:** Randomly shuffles tokens in the input and trains the model to generate them in the correct original order.

### Sentence & Document-Level Objectives

- **Next Sentence Prediction (NSP):** Feeds two sentences and predicts whether the second sentence logically follows the first in the original text (BERT pre-training).
- **Sentence Comparison:** Computes sentence embeddings to output a similarity score determining semantic equivalence or entailment (used in bi-encoders like SBERT).
- **Sentence Reordering:** Shuffles the order of full sentences in a paragraph; the model must reconstruct the correct logical paragraph flow.
- **Document Rotation:** Selects a random token as a rotation point and wraps the beginning of the text to the end; the model must locate the true start and restore the document.
