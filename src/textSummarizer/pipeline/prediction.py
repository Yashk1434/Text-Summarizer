from textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer, pipeline
import torch
import re


class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

    def predict(self, text):
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)

        # Count the number of words in the original text
        word_count = len(text.split())

        # Set the target word count to approximately one-third of the original word count
        target_word_count = max(50, word_count // 3)  # Ensure at least 50 words for the summary

        # Check for GPU availability
        device = 0 if torch.cuda.is_available() else -1

        # Set generation parameters
        gen_kwargs = {
            "length_penalty": 1.5,  # Adjust for balance between concise and detailed summaries
            "num_beams": 8,  # More beams for better quality
            "min_length": max(100, target_word_count // 2),  # Minimum length is half of target
            "max_length": target_word_count + 20  # Allow flexibility in max length
        }

        # Initialize the pipeline with correct device
        pipe = pipeline("summarization", model="csebuetnlp/mT5_multilingual_XLSum", device=device)

        print("Original Text:")
        print(text)

        # Generate the summary
        output = pipe(text, **gen_kwargs)[0]["summary_text"]

        print("\nModel Summary:")
        print(output)

        # Ensure the output ends with a complete sentence
        output = self.ensure_complete_sentence(output)

        return output

    def ensure_complete_sentence(self, summary):
        # Use regex to find the last sentence-ending punctuation
        last_sentence_end = re.search(r'[.!?]', summary)
        if last_sentence_end:
            # Trim the summary to end at the last sentence-ending punctuation
            return summary[:last_sentence_end.end()].strip()
        return summary.strip()  # Return as is if no sentence-ending punctuation is found
