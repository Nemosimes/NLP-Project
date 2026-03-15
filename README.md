# MS1 Technical Report

## Design Choices

The main design steps were:

1. Load transcripts and QA pairs
2. Perform exploratory data analysis
3. Detect noise and linguistic patterns
4. Apply Arabic normalization
5. Attach transcript context to QA pairs
6. Tokenize text
7. Build vocabulary and token IDs
8. Apply padding for neural models

---

## Exploratory Data Analysis

We analyzed:
- Question and answer length distribution
- Transcript length distribution
- Vocabulary size
- Most frequent words
- Most common bigrams
- Repeated question templates

We also checked linguistic properties such as:
- English words inside Arabic text 
- Egyptian dialect words
- Narrative markers used in speech
- Named entities such as scientists and historical figures
- Timestamp markers inside transcripts

These analyses confirmed that the dataset is realistic but noisy, and requires normalization before modeling.

---

## Normalization Design

A custom normalization function was implemented using regular expressions.  
The normalization was applied in a fixed order to keep the output consistent.

The steps used were:

1. Remove timestamps
2. Remove diacritics
3. Remove kashida (tatweel)
4. Normalize alef variants to ا
5. Apply spelling corrections using a dictionary
6. Normalize final letters (ى → ي, ؤ → و, ئ → ي, ة → ه)
7. Collapse repeated punctuation
8. Remove non-Arabic characters
9. Normalize whitespace

We added small spelling correction dictionary to fix common mistakes such as:
- هاذا → هذا
- لاكن → لكن
- ان شاء الله → إن شاء الله

This approach reduces vocabulary size while keeping the meaning of the text.

---

## Output Analysis

After normalization, the dataset became more consistent.

We observed that :
- Vocabulary size decreased after normalization
- Noise symbols were removed
- Tokenization became easier
- Questions and answers still matched the transcripts

We also attached the full transcript as context for each QA pair to prepare the data for sequence-to-sequence or retrieval models.

After that, the text was tokenized and converted to IDs using a vocabulary map.
Two special tokens were added:
- `<PAD>` for padding
- `<UNK>` for unknown words

Padding was applied using fixed lengths:
- Context length = 7500
- Answer length = 10

This ensures that the data can be used directly in neural networks.

---

## Reasoning Behind Padding Length

The context length was set to 7500 because transcript sizes are large and we need enough space to keep most of the text.  
The answer length was set to 10 because answers are usually short spans.

Using fixed padding makes the input compatible with neural models.

---

## Limitations

The framework works well, but it has some limitations.

First, normalization rules are rule-based and may not handle all dialect variations.  
Second, some English words and informal expressions still remain.  
Third, large context padding increases memory usage and may slow down training.  
Fourth, spelling correction uses a small dictionary, so it does not fix all errors.

Another limitation is that answers must remain exact spans from transcripts, so aggressive cleaning cannot be applied.
