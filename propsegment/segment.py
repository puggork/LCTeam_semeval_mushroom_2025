import argparse
import pandas as pd
import json

'''
This program splits model output into propositions via the deletion of ngrams.
You can specify the file to segment, where to save it, and the length of the ngram span
through command line arguments:
@arg -f, --file: Specifies the file to segment into propositions
@arg -o, --output_file: Specifies the output file
@arg -n, --ngram: Specifies the length of the ngram span
'''

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', default='../dataset/train/mushroom.en-train_nolabel.v1.jsonl', type=str,
                        help="File to segment into propositions")
    parser.add_argument('-o', '--output_file', default='output', type=str,
                        help="Output file")
    parser.add_argument('-n', '--ngram', default=1, type=int,
                        help="Specify ngram length")
    args = parser.parse_args()
    return args


def load_json(filename):
    with open(filename,'r') as f:
        df = pd.read_json(f, lines=True)
    return df.to_dict(orient='records')


# takes in item as list, and deletes ngram spans of the specified length
# returns list with all the possible segmentations
def segment_item(item, ngram_length):
    propositions = []

    for start_id in range(len(item)):
        prop = item.copy()
        end_id = start_id+ngram_length
        if len(prop)>end_id: #normal case - end of span is within sentence bounds
            del prop[start_id:end_id]
        else: #ngram span exceeds sentence boundary
            del prop[start_id:]
        propositions.append(prop)
    
    return propositions


# takes in data as dict and desired length of ngrams 
# returns dict of lists of all possible output propositions
def segment_propositions(data, ngram_length):
    segmented_data = []
    for line in data:
        segmented_line = {'lang':line['lang'],
                          'model_id':line['model_id'],
                          'model_input':line['model_input'],
                          'model_output_text':segment_item(line['model_output_text'].split(' '), ngram_length), # very primitive tokenization, could do better esp. for chinese
                          'model_output_logits':segment_item(line['model_output_logits'], ngram_length),
                          'model_output_tokens':segment_item(line['model_output_tokens'], ngram_length)
                          }
        segmented_data.append(segmented_line)
    
    return segmented_data


# Parse the input arguments
args = create_arg_parser()

data = load_json(args.file)
segmented_data = segment_propositions(data,args.ngram)

with open(f"{args.output_file}_{args.ngram}grams.jsonl",'w') as output:
    json.dump(segmented_data, output)