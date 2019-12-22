import json
import pandas as pd

train_file = '/Users/sachin/Data/Kaggle/tfqa/tfqa_data/simplified-nq-train.jsonl'

chunk_size = 1

for chunk in pd.read_json(train_file, lines=True, chunksize=chunk_size):
    #print(chunk.columns)
    print(chunk['document_text'])
    print(chunk['long_answer_candidates'])
    print(chunk['question_text'])
    break
