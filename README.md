# mushroom_lct_2024
Submission for SemEval-2025 Task-3 — Mu-SHROOM.

Task: [link](https://helsinki-nlp.github.io/shroom/)

# Pipeline description (from Araya)

1. Generate Wikipedia Summary: Generate a summary from Wikipedia based on self-checking GPT API. Choose one summary out of 3 using self-checking.

2. Correct LLM Output: Correct the LLM's output based on the Wikipedia summary using GPT API.

3. Identify Hallucinated and Overgenerated Text Chunks: Identify hallucinated and overgenerated text chunks by extracting keywords (named entities, nouns, adjectives) from both the corrected LLM output and the original LLM output.

4. Hard Label Annotation: Perform hard label annotation based on text spans of the detected text chunks.

# Detailed step-by-step on how the files are used and manipulated

All the files are in the folder 'full_baseline_20250102/.'.

* The notebook 'Automatic_labeler_new_version.ipynb' performs steps 1 and 2 of the pipeline and uses the unlabelled train data from the file 'mushroom.en-train_nolabel.v1.jsonl'. You have to use your own API key from OpenAI. It won't work if you're using the Free plan. But apparently, this notebook generates the new columns for data from the file 'en_train_labeled.jsonl', which are: 'generated_summary' and 'corrected_output'. The data is saved to 'en_train_automatically_labeled_output_complete.jsonl'. (Why does this file have "hard_labels??? It is NOT supposed to!")

* The notebook 'pseudo_labeling.ipynb' performs step 3 of the pipeline. It takes the data from 'en_train_automatically_labeled_output_complete.jsonl'. Using spacy, nltk and RAKE, it adds three new columns to the data ('updated_generated_text', 'hallucinated_clauses' and 'clause_similarity'), as well as hard labels. The data is saved to the file named 'en_train_new1.jsonl'.

* The notebook 'Baseline_v3.ipynb' performs step 4 of the pipeline. It takes the labelled train data from 'en_train_new1.jsonl' and validation data from 'mushroom.en-val.v2.jsonl'. It removes the responses from the labelled train data starting with "I'm sorry". It trains a 'roberta-base' model, with evaluation of the generated soft and hard labels.