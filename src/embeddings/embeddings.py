"""
Email embedding module to generate embeddings for emails
"""
from typing import List
import torch
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
import torch.nn.functional as F

class EmailEmbedder:
    def __init__(self, model_name: str = "infly/inf-retriever-v1"):
        """Initialize email embedder.
        
        Args:
            model_name: Name of the model to use
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        self.model = AutoModel.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map={"": 0},  # load everything to GPU 0
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

    def embed_emails(self, emails: List[str], batch_size: int = 2) -> torch.Tensor:
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

            with torch.no_grad():
                model_output = self.model(**encoded_input)

            embeddings = self.mean_pool(model_output, encoded_input["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0)

    def embed_query(self, query: str) -> torch.Tensor:
        """Generate embedding for a search query.
        
        Args:
            query: Search query
            
        Returns:
            Query embedding tensor
        """
        # TODO: Implement query embedding
        pass
