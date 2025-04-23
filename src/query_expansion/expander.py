from typing import List
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from src.utils import set_global_seed

class QueryExpander:
    def __init__(self, model_name: str = "eugenesiow/bart-paraphrase", seed: int = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name).to(self.device)
        if seed is not None:
            set_global_seed(seed)
    def expand(self, query: str, num_variants: int = 10) -> List[str]:
        input_ids = self.tokenizer(query, return_tensors="pt", truncation=True, padding="longest").input_ids.to(self.device)

        outputs = self.model.generate(
            input_ids,
            max_length=128,
            num_beams=1, 
            num_return_sequences=num_variants * 2,
            do_sample=True,              
            top_k=30,                    
            top_p=0.9,                  
            temperature=0.95,             
            no_repeat_ngram_size=3,
            early_stopping=False
        )
        paraphrases = [self.tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]
        paraphrases = [p for p in paraphrases if p != query]
        paraphrases += [query]
        paraphrases = [p.lower() for p in paraphrases]
        paraphrases = list(set(paraphrases)) 
        return paraphrases

    