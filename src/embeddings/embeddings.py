"""
Email embedding module to generate embeddings for emails
"""
from typing import List
import torch
from transformers import AutoTokenizer, AutoModel

class EmailEmbedder:
    def __init__(self, model_name: str = "Linq-AI-Research/Linq-Embed-Mistral"):
        """Initialize email embedder.
        
        Args:
            model_name: Name of the model to use
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

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

    def embed_emails(self, emails: List[str]) -> torch.Tensor:
        """Generate embeddings for a list of emails.
        
        Args:
            emails: List of email texts
            
        Returns:
            Tensor of email embeddings
        """
        encoded_input = self.tokenizer(emails, padding=True, truncation=True, return_tensors="pt")
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

        with torch.no_grad():
            model_output = self.model(**encoded_input)
    
        embeddings = self.mean_pool(model_output, encoded_input["attention_mask"])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def embed_query(self, query: str) -> torch.Tensor:
        """Generate embedding for a search query.
        
        Args:
            query: Search query
            
        Returns:
            Query embedding tensor
        """
        # TODO: Implement query embedding
        pass 