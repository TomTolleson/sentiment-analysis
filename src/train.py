import os
from transformers import AutoModelForSequenceClassification
from huggingface_hub import push_to_hub
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer
)
from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

def train_model():
    # Login to Hugging Face
    login(token=os.getenv('HUGGINGFACE_TOKEN'))

    # Load model and tokenizer
    model_name = "distilbert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=3
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load dataset
    dataset = load_dataset("financial_phrasebank", "sentences_allagree")

    # Check available splits
    print(dataset)

    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"], 
            padding="max_length", 
            truncation=True
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Update the eval_dataset to use the correct split
    # Check available splits in the tokenized dataset
    eval_dataset_key = "test"  # Change this if the dataset does not have a 'test' split
    if eval_dataset_key not in tokenized_dataset:
        eval_dataset_key = "validation"  # Fallback to 'validation' if 'test' is not available
    if eval_dataset_key not in tokenized_dataset:
        eval_dataset_key = "train"  # Fallback to 'train' if neither 'test' nor 'validation' is available
    if eval_dataset_key not in tokenized_dataset:
        raise ValueError("No valid splits available in the dataset.")

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./models",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        push_to_hub=True,
        hub_model_id=f"{os.getenv('HUGGINGFACE_USERNAME')}/sentiment-analysis"
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset[eval_dataset_key]
    )

    # Train and push to hub
    trainer.train()
    trainer.push_to_hub()

if __name__ == "__main__":
    train_model()


model.push_to_hub("tangohoteltango/sentiment-analysis")
tokenizer.push_to_hub("tangohoteltango/sentiment-analysis")