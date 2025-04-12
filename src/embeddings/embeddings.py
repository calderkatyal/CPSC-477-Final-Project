"""
Email embedding module to generate embeddings for emails
"""
import os
from typing import List
import torch
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
import torch.nn.functional as F

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,expandable_segments:True"

class EmailEmbedder:
    def __init__(self, model_name: str = "infly/inf-retriever-v1-1.5b"):
        """Initialize email embedder.
        
        Args:
            model_name: Name of the model to use
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=False
        )

        self.model = AutoModel.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map={"": self.device},
            trust_remote_code=True
        )

    @staticmethod
    def mean_pool(model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply mean pooling to get sentence embeddings

        Args:
            model_output: Model output tensor
            attention_mask: Attention mask tensor
        
        Returns:
            Mean pooled sentence embeddings
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.inference_mode()
    def embed_emails(self, emails: List[str], batch_size) -> torch.Tensor:
        """Generate embeddings for a list of emails in batches.
        
        Args:
            emails: List of email texts
            batch_size: Number of emails per batch
            
        Returns:
            Tensor of email embeddings
        """
        all_embeddings = []

        for i in range(0, len(emails), batch_size):
            batch = emails[i:i + batch_size]
            encoded_input = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

            model_output = self.model(**encoded_input)

            embeddings = self.mean_pool(model_output, encoded_input["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())

            # Free up memory
            del model_output, encoded_input, embeddings
            torch.cuda.empty_cache()

            # Optional: log memory
            # print(f"[Batch {i}] Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB | Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

        return torch.cat(all_embeddings, dim=0)

    @torch.inference_mode()
    def embed_query(self, query: str) -> torch.Tensor:
        """Generate embedding for a search query.
        
        Args:
            query: Search query
            
        Returns:
            Query embedding tensor
        """
        encoded_input = self.tokenizer(query, return_tensors="pt")
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        model_output = self.model(**encoded_input)
        embeddings = self.mean_pool(model_output, encoded_input["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    