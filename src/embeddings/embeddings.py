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
    def __init__(self, big_model: bool = False):
        """Initialize email embedder.
        
        Args:
            use_full_precision: Whether to use full precision 'infly/inf-retriever-v1' model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if big_model:
            model_name = "infly/inf-retriever-v1"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                model_name,
                device_map={"": self.device},
                trust_remote_code=True
            )
        else:
            model_name = "infly/inf-retriever-v1-1.5b"
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
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.inference_mode()
    def embed_emails(self, emails: List[str], batch_size) -> torch.Tensor:
        all_embeddings = []

        for i in range(0, len(emails), batch_size):
            batch = emails[i:i + batch_size]
            encoded_input = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

            model_output = self.model(**encoded_input)

            embeddings = self.mean_pool(model_output, encoded_input["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())

            del model_output, encoded_input, embeddings
            torch.cuda.empty_cache()

        return torch.cat(all_embeddings, dim=0)

    @torch.inference_mode()
    def embed_query(self, query: str) -> torch.Tensor:
        encoded_input = self.tokenizer(query, return_tensors="pt")
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        model_output = self.model(**encoded_input)
        embeddings = self.mean_pool(model_output, encoded_input["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
