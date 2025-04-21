from typing import List
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

class QueryExpander:
    def __init__(self, model_name: str = "eugenesiow/bart-paraphrase"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name).to(self.device)
    def expand(self, query: str, num_variants: int = 3) -> List[str]:
        input_ids = self.tokenizer(query, return_tensors="pt", truncation=True, padding="longest").input_ids.to(self.device)

        outputs = self.model.generate(
            input_ids,
            max_length=128,
            num_beams=num_variants * 2,
            num_return_sequences=num_variants,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
        )

        paraphrases = [self.tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]
        paraphrases = [p for p in paraphrases if p != query]
        paraphrases = list(set(paraphrases)) 
        return paraphrases

if __name__ == "__main__":
    query_expander = QueryExpander()
    query = input("Enter your query: ")
    variants = query_expander.expand(query)
    print("Generated Variants:")
    for i, variant in enumerate(variants):
        print(f"{i + 1}: {variant}")
    print("\n")

    