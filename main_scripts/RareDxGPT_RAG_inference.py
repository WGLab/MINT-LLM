import json
import os
import re
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoProcessor
import datasets
from datasets import load_from_disk, load_dataset
import wandb
from accelerate import PartialState
from peft import AutoPeftModelForCausalLM
import sys
sys.path.append(os.path.abspath('/home/wangz12/projects/RareDxGPT/utils'))
from set_seed import *
from util_llama3 import *
from disease_list_extract import *
from external_analysis_util import *
sys.path.append(os.path.abspath('/home/wangz12/projects/RareDxGPT/AutoEvaluator'))
from AutoEvaluator import *
from EvaluatorProcessor import *
import pandas as pd
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
    get_response_synthesizer,
    ChatPromptTemplate,
)
from huggingface_hub import login
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

def parse_args():

    parser = argparse.ArgumentParser(
        description="RareDxGPT-orpo"
    )
    
    parser.add_argument("--batch_size",
                        type=int,
                        default=8,
                        help="Batch size for training"
                        )
    
    parser.add_argument("--ratio",
                        type=float,
                        default=0.3,
                        help="Train test split ratio"
                        )

    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed for train test split")

    parser.add_argument("--disease",
                        type=str,
                        default='bws',
                        help='Disease name for external dataset')

    parser.add_argument("--peft_model_id",
                        type=str,
                        default="x")
    return parser.parse_args()




def setup_ddp():
    dist.init_process_group(backend='nccl')

def cleanup_ddp():
    dist.destroy_process_group()

def setup_rag_system(model, tokenizer):
    # Configure LlamaIndex settings with new approach
    Settings.llm = HuggingFaceLLM(
        model=model,
        tokenizer=tokenizer,
        context_window=3900,
        max_new_tokens=256,
        generate_kwargs={
            'do_sample': True,
            'temperature': 1.0,
            "top_k": 50,
            "top_p": 0.9
        },
        device_map='auto'
    )
    
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    Settings.node_parser = SentenceSplitter(
        chunk_size=256,
        chunk_overlap=20,
        paragraph_separator="\n\n",
        include_metadata=True
    )
    
    return Settings

def create_rag_index(data_dir):
    documents = SimpleDirectoryReader(
        input_dir=data_dir,
        filename_as_id=True
    ).load_data()
    
    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True
    )
    
    return index

def main():
    args = parse_args()
    set_seed(args.seed)
    login(token ="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")
    os.environ['HF_HOME'] = '/tmp'
    # Model Setup
    peft_model_id = os.path.join("/home/wangz12/projects", args.peft_model_id)
    peft_model_id = peft_model_id
    local_rank = int(os.environ['LOCAL_RANK'])
    # torch.cuda.set_device(local_rank)
    # setup_ddp()
    model = AutoModelForCausalLM.from_pretrained(
        peft_model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    # model = AutoPeftModelForCausalLM.from_pretrained(
    #     peft_model_id,
    #     device_map="auto",
    #     torch_dtype=torch.bfloat16
    # )
    tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
    model.resize_token_embeddings(len(tokenizer))
    
    # Dataset Loading
    total_train = load_dataset("csv", data_files="/home/wangz12/projects/RareDxGPT/reference_data/total_train.csv")
    disease_name = pd.read_csv("/home/wangz12/projects/RareDxGPT/reference_data/disease_name_full.csv")
    reference_list = list(disease_name.Name)
    test_dataset = load_from_disk(f'/home/wangz12/projects/RareDxGPT/datasets/phen2gene_gmdb_dataset')
    # test_dataset = test_dataset.rename_column("Prompt", "prompt")
    # ground_truth_list = test_dataset['Response']  
    ground_truth_list = test_dataset['disease']
    # RAG System Setup
    settings = setup_rag_system(model, tokenizer)
    index = create_rag_index("/home/wangz12/projects/RareDxGPT/datasets/rag_data")
    
    chat_text_qa_messages = [
        (
            "user",
            """You are a genetic counselor. Your task is to identify potential rare diseases based on given phenotypes. Follow the output format precisely.
            
            Context:
            {context_str}
            
            Question:
            {query_str}
            
            Based on this information, provide: A numbered list of EXACTLY 10 potential rare diseases.\n\nUse EXACTLY this format:\n\nPOTENTIAL_DISEASES:\n1. 'Disease1'\n2. 'Disease2'\n3. 'Disease3'\n4. 'Disease4'\n5. 'Disease5'\n6. 'Disease6'\n7. 'Disease7'\n8. 'Disease8'\n9. 'Disease9'\n10. 'Disease10'\n\nEnsure all disease names are in single quotes, properly capitalized, and there are exactly 10 in the list. Do not deviate from this format or add any explanations.""",
        )
    ]
    
    # Set up query engine
    chat_template = ChatPromptTemplate.from_messages(chat_text_qa_messages)
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=3
    )
    
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        response_synthesizer=get_response_synthesizer(
            text_qa_template=chat_template,
            response_mode="compact"
        ),
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=0.7)
        ]
    )
    
    # Query Processing
    results = []
    for query in tqdm(test_dataset['prompt']):
        try:
            response = query_engine.query(query)
            if hasattr(response, 'source_nodes'):
                print(f"Retrieved {len(response.source_nodes)} source nodes")
                for node in response.source_nodes:
                    print(f"Relevance score: {node.score:.3f}")
            results.append(response.response)
        except Exception as e:
            print(f"Error processing query: {e}")
            results.append(None)
    
    # Post-processing and Evaluation
    inference_list = [extraction3(text) if text else None for text in results]
    
    processor = EvaluationProcessor(reference_list)
    eval_results = processor.evaluate_samples(
        inference_list,
        ground_truth_list,
        k=10,
        q=10,
        lambda_weight=1.0,
        calc_coverage=False,
        calc_avoidance=False,
        calc_car=False
    )
    print(eval_results)
    
    # Analysis of results
    stats = analyze_disease_rankings(inference_list)
    
    # Print statistics
    for position in sorted(stats.keys()):
        print(f"\nPosition {position} disease counts:")
        if stats[position]:
            for disease, count in sorted(stats[position].items(), key=lambda x: x[1], reverse=True):
                print(f"  {disease}: {count}")
        else:
            print("  No diseases found at this position")
    
    all_diseases = {}
    for pos_stats in stats.values():
        for disease, count in pos_stats.items():
            all_diseases[disease] = all_diseases.get(disease, 0) + count
    
    print("\nTotal occurrences across positions:")
    for disease, count in sorted(all_diseases.items(), key=lambda x: x[1], reverse=True):
        print(f"  {disease}: {count}")
    
    cleanup_ddp()

if __name__ == "__main__":
    main()