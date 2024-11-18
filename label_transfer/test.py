#!/usr/bin/env python3

import os
import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Function to load translation model dynamically based on language code
def load_model(lang_code, device=0):
    if lang_code != 'en':
        try:
            # Constructs the model name based on the language code
            model_name = f"Helsinki-NLP/opus-mt-{lang_code}-en"
            return pipeline(f"translation_{lang_code}_to_en", model=model_name, device=device)
        except Exception as e:
            print(f"Error loading model for {lang_code}: {e}")
            return None

# Load optimized embedding model (Sentence-Transformer)
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device=device)

# Function to get embeddings for texts
def get_embeddings(texts):
    return embedding_model.encode(texts)

# Function to find similar spans between original and back-translated text
def find_similar_spans(original_text, translated_text, spans, length_variation=5):
    similar_spans = []
    
    for start, end in spans:
        # Extract the original span text based on character indices
        original_span_text = original_text[start:end]
        original_span_embedding = get_embeddings([original_span_text])[0]
        
        # Initialize tracking for best match
        max_similarity = 0
        best_span = [0, 0]
        
        # Try a range of candidate spans in the translated text around the original span length
        original_span_length = end - start
        for candidate_start in range(len(translated_text)):
            for candidate_length in range(original_span_length - length_variation, original_span_length + length_variation + 1):
                # Ensure candidate span stays within bounds
                candidate_end = candidate_start + candidate_length
                if candidate_end > len(translated_text):
                    continue
                
                # Candidate span text and embedding
                candidate_span_text = translated_text[candidate_start:candidate_end]
                candidate_span_embedding = get_embeddings([candidate_span_text])[0]
                
                # Calculate similarity
                similarity = cosine_similarity([original_span_embedding], [candidate_span_embedding])[0][0]
                
                # Update best match if this candidate is more similar
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_span = [candidate_start, candidate_end]
        
        # Adjust best_span indices to exclude leading/trailing spaces
        best_start, best_end = best_span
        while best_start < best_end and translated_text[best_start] == ' ':
            best_start += 1
        while best_end > best_start and translated_text[best_end - 1] == ' ':
            best_end -= 1
        
        # Append the best match found for this original span
        similar_spans.append([best_start, best_end])
    
    return similar_spans

def load_jsonl_file_to_records_(filename):
    try:
        df = pd.read_json(filename, lines=True)
        df['text_len'] = df.model_output_text.apply(len)
        return df.sort_values('id').to_dict(orient='records')
    except ValueError as e:
        print(f"Error reading {filename}: {e}")
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                try:
                    pd.read_json(line, lines=True)
                except ValueError as line_error:
                    print(f"Invalid JSON on line {i+1} in {filename}: {line.strip()} - {line_error}")
        return []  # Return an empty list if file loading fails


# Function to load and concatenate multiple .jsonl files from a directory
def load_and_concatenate_jsonl_files(directory):
    all_records = []

    # Loop through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.jsonl'):
            filepath = os.path.join(directory, filename)
            records = load_jsonl_file_to_records_(filepath)
            all_records.extend(records)
    
    # Convert the list of records into a single DataFrame
    concatenated_df = pd.DataFrame(all_records)
    return concatenated_df

# Example usage:
# Specify the directory containing the .jsonl files
directory_path = 'val'

# Load and concatenate the JSONL files
combined_df = load_and_concatenate_jsonl_files(directory_path)
print(f"total_data: {len(combined_df)}")

# Prepare lists to store results
translated_texts = []
translated_hard_labels = []
#combined_df = combined_df[:1]
# Loop through each record, load the appropriate model, and perform translation
for index, row in combined_df.iterrows():
    lang_code = row['lang'].lower()
    original_text = row['model_output_text']
    spans = row['hard_labels']
    
    # Dynamically load the model for the current language
    model = load_model(lang_code, device=device) 
    if model is None:
        continue  # Skip if model loading failed
    
    # Perform translation to English
    translated_text = model(original_text)[0]['translation_text']
    
    # Find the most similar spans in the translated text
    translated_hard_label = find_similar_spans(original_text, translated_text, spans)
    
    # Store the results
    translated_texts.append(translated_text)
    translated_hard_labels.append(translated_hard_label)

# Add translated results to DataFrame
combined_df['translated_text'] = translated_texts
combined_df['translated_hard_labels'] = translated_hard_labels

# Save to CSV file

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.realpath(__file__))

# Define the path for the Output directory
output_dir = os.path.join(script_dir, 'Output')

# Check if the directory exists, if not, create it
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

# Save the final DataFrame to a CSV file in the Output directory
output_csv = os.path.join(output_dir, 'output.csv')  
combined_df.to_csv(output_csv, index=False)

# Print the path where the file was saved
print(f"Data saved to {output_csv}")

