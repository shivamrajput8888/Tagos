from transformers import DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

tokenizer.save_pretrained("saved_model")

print("Tokenizer regenerated successfully!")