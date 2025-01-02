# mushroom_lct_2024
Submission for SemEval-2025 Task-3 — Mu-SHROOM.

Task: [link](https://helsinki-nlp.github.io/shroom/)

# Pipeline description (from Araya)

1. Generate Wikipedia Summary: Generate a summary from Wikipedia based on self-checking GPT API. Choose one summary out of 3 using self-checking.

2. Correct LLM Output: Correct the LLM's output based on the Wikipedia summary using GPT API.

3. Identify Hallucinated and Overgenerated Text Chunks: Identify hallucinated and overgenerated text chunks by extracting keywords using the RAKE method from NLTK from both the corrected LLM output and the original LLM output.

4. Hard Label Annotation: Perform hard label annotation based on text spans of the detected text chunks.

# Detailed step-by-step on how the files are used and manipulated

All the files are in the folder 'full_baseline_20250102/.'.

* The notebook 'Automatic_labeler_new_version.ipynb' performs steps 1 and 2 of the annotation pipeline and uses the unlabeled training data from the file 'mushroom.en-train_nolabel.v1.jsonl'. You have to use your own API key from OpenAI; it won't work if you're using the Free plan. This notebook generates new columns, which are 'generated_summary' and 'corrected_output'. The data is saved to 'en_train_automatically_labeled_output_complete.jsonl'. (Why does this file have "hard_labels"? It is NOT supposed to!) (Answer: Yes, it is not supposed to appear, but I have used it from my previous trials using another approach, so instead of regenerating 'generated_summary' and 'corrected_output' as it is based on the API. The existing hard label value will be overridden with the new one.)

* The notebook 'pseudo_labeling.ipynb' performs step 3 & 4 of the annotation pipeline. It takes the data from 'en_train_automatically_labeled_output_complete.jsonl'. Using spacy, nltk and RAKE, it adds three new columns to the data ('updated_generated_text', 'hallucinated_clauses', and 'clause_similarity') and hard labels. The data is saved to the file named 'en_train_labeled.jsonl'.

* The notebook 'Baseline_v3.ipynb' performs supervised learning by taking labeled training data for token classification to identify hallucinated and overgenerated text spans. It takes the labeled training data from 'en_train_new1.jsonl' and validation data from 'mushroom.en-val.v2.jsonl'. Noisy data filtration removes the model responses containing "I'm sorry" from the labeled training data. It trains a Roberta-base model by aligning hard labels with token offsets. During inference, probabilistic predictions are used as soft labels, and hard labels are computed from the soft labels using a threshold value of 0.5.
