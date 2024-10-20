from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from evaluate import load as load_metric
import torch
import pandas as pd
from tqdm import tqdm
from textSummarizer.entity import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        """Split the dataset into smaller batches that we can process simultaneously.
        Yield successive batch-sized chunks from list_of_elements."""
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i: i + batch_size]

    def calculate_metric_on_test_ds(self, dataset, metric, model, tokenizer, 
                                    batch_size=16, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
                                    column_text="article", column_summary="highlights"):
        """Calculates metrics on the test dataset in batches."""
        model.to(device)  # Move model to the correct device

        article_batches = list(self.generate_batch_sized_chunks(dataset[column_text], batch_size))
        target_batches = list(self.generate_batch_sized_chunks(dataset[column_summary], batch_size))

        for article_batch, target_batch in tqdm(
                zip(article_batches, target_batches), total=len(article_batches)):
            
            # Tokenize inputs and move tensors to the correct device
            inputs = tokenizer(article_batch, max_length=1024, truncation=True, 
                               padding="max_length", return_tensors="pt").to(device)
            
            # Generate summaries and ensure they are on the same device
            with torch.no_grad():  # No need for gradients during evaluation
                summaries = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"], 
                    length_penalty=0.8, 
                    num_beams=8, 
                    max_length=128
                )
            
            # Decode the generated summaries
            decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, 
                                                  clean_up_tokenization_spaces=True) 
                                 for s in summaries]
            
            decoded_summaries = [d.replace("", " ") for d in decoded_summaries]
            
            # Add batch to the metric for evaluation
            metric.add_batch(predictions=decoded_summaries, references=target_batch)

        # Compute and return the final score
        score = metric.compute()
        return score

    def evaluate(self):
        """Evaluates the model on the test dataset and computes the ROUGE score."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "csebuetnlp/mT5_multilingual_XLSum"
        
        # Load the tokenizer and model, move model to the correct device
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
       
        # Load the dataset
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        # Load ROUGE metric
        rouge_metric = load_metric('rouge')

        # Evaluate the model and calculate ROUGE scores on a sample of the dataset
        score = self.calculate_metric_on_test_ds(
            dataset_samsum_pt['test'][0:10], 
            rouge_metric, 
            model, 
            tokenizer, 
            batch_size=2, 
            column_text='dialogue', 
            column_summary='summary'
        )

        # ROUGE score dictionary
        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        rouge_dict = {rn: score[rn].mid.fmeasure if hasattr(score[rn], 'mid') else score[rn] for rn in rouge_names}

        # Save scores to a CSV file
        df = pd.DataFrame(rouge_dict, index=['mT5_multilingual_XLSum'])
        df.to_csv(self.config.metric_file_name, index=False)
