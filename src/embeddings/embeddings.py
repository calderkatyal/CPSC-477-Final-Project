"""
Email embedding module to generate embeddings for emails
"""
import os
from typing import List
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from src.utils import set_global_seed

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,expandable_segments:True"

class EmailEmbedder:
    def __init__(self, seed: int = None):
        """Initialize email embedder.
        
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_name = "infly/inf-retriever-v1-1.5b"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        self.model = AutoModel.from_pretrained(
            model_name,
            device_map={"": self.device},
            trust_remote_code=True
        )
        if seed is not None:
            set_global_seed(seed)

    @staticmethod
    def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


    @torch.inference_mode()
    def embed_emails(self, emails: List[str], batch_size) -> torch.Tensor:
        all_embeddings = []

        for i in range(0, len(emails), batch_size):
            batch = emails[i:i + batch_size]
            encoded_input = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

            model_output = self.model(**encoded_input)

            embeddings = self.last_token_pool(model_output.last_hidden_state, encoded_input["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())

            del model_output, encoded_input, embeddings
            torch.cuda.empty_cache()

        return torch.cat(all_embeddings, dim=0)

    @torch.inference_mode()
    def embed_query(self, queries: List[str]) -> torch.Tensor:
        task = "Given an email search query, retrieve the most relevant emails"

        prompts = [f"Instruct: {task}\nQuery: {q}" for q in queries]

        encoded_input = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=8192)
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

        model_output = self.model(**encoded_input)
        embeddings = self.last_token_pool(model_output.last_hidden_state, encoded_input["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

