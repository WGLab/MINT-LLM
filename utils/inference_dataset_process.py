import pandas as pd
import datasets
from datasets import load_from_disk, load_dataset

total_train = load_dataset("csv", data_files="/home/wangz12/projects/RareDxGPT/reference_data/total_train.csv")
disease_name = pd.read_csv("/home/wangz12/projects/RareDxGPT/reference_data/disease_name_full.csv")
disease_name = list(disease_name.Name)
reference_list = disease_name
full_dataset = total_train['train']
# load into pipeline
test_dataset_dict = load_from_disk("/home/wangz12/projects/RareDxGPT/datasets/orpo_dpo_dataset_cask10_10")
# test_dataset_dict = test_dataset_dict.remove_columns('image_id')
test_dataset = test_dataset_dict['test']
def format_data(example):
    return {"messages": [
    {"role": "system", "content": "You are a genetic counselor. Your task is to identify potential rare diseases based on given phenotypes. Follow the output format precisely."},
    {"role": "user", "content": f"{example['prompt']}\n\nBased on this information, provide a numbered list of EXACTLY 10 potential rare diseases.\n\nUse EXACTLY this format:\n\nPOTENTIAL_DISEASES:\n1. 'Disease1'\n2. 'Disease2'\n3. 'Disease3'\n4. 'Disease4'\n5. 'Disease5'\n6. 'Disease6'\n7. 'Disease7'\n8. 'Disease8'\n9. 'Disease9'\n10. 'Disease10'\n\nEnsure all disease names are in single quotes, and there are exactly 10 in the list. Do not deviate from this format or add any explanations."},
    ]}
test_dataset = test_dataset.map(format_data)
def format_chat_template(row):
    row["messages"] = tokenizer.apply_chat_template(row["messages"], tokenize=False)
    return row
test_dataset = test_dataset.map(format_chat_template)
############Ground Truth List Filtering#####################
dataset1 = full_dataset
dataset2 = test_dataset
dataset1_df = dataset1.to_pandas()
dataset2_df = dataset2.to_pandas()

dataset1_df['image_id'] = dataset1_df['image_id'].astype(str)
dataset2_df['image_id'] = dataset2_df['image_id'].astype(str)

merged_df = pd.merge(dataset2_df[['image_id']], dataset1_df[['image_id', 'Response']], on='image_id', how='left')

ground_truth_list = merged_df['Response'].tolist()