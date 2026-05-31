import argparse
import json
import os
from typing import Dict, List

from dotenv import load_dotenv
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


class FineTuneDataset(Dataset):
    def __init__(self, encodings: Dict[str, List[int]]) -> None:
        self.encodings = encodings

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }


def _format_record(record: Dict) -> str:
    if "text" in record and isinstance(record["text"], str):
        return record["text"].strip()

    question = record.get("question") or record.get("input") or ""
    answer = record.get("answer") or record.get("output") or ""
    instruction = record.get("instruction") or ""

    return (
        f"Instruction: {instruction}\n"
        f"Question: {str(question).strip()}\n"
        f"Answer: {str(answer).strip()}"
    ).strip()


def load_training_texts(data_path: str) -> List[str]:
    texts: List[str] = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = _format_record(record)
            if text:
                texts.append(text)

    if not texts:
        raise ValueError("No training examples found in dataset.")
    return texts


def train_finetuned_model() -> None:
    load_dotenv()

    base_model = os.getenv("FINETUNE_BASE_MODEL", "distilgpt2")
    data_path = os.getenv("FINETUNE_DATA_PATH", "finetune_data.jsonl")
    output_dir = os.getenv("FINETUNE_OUTPUT_DIR", "./finetuned_model")

    max_length = int(os.getenv("FINETUNE_MAX_LENGTH", "512"))
    epochs = float(os.getenv("FINETUNE_EPOCHS", "1"))
    batch_size = int(os.getenv("FINETUNE_BATCH_SIZE", "2"))
    learning_rate = float(os.getenv("FINETUNE_LEARNING_RATE", "2e-5"))

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Training data not found at '{data_path}'. Provide a JSONL file with text or question/answer records."
        )

    texts = load_training_texts(data_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model)
    model.config.pad_token_id = tokenizer.pad_token_id

    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    dataset = FineTuneDataset(encodings)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Fine-tuned model saved to: {output_dir}")


def answer_with_finetuned_model(question: str) -> str:
    load_dotenv()

    model_dir = os.getenv("FINETUNE_OUTPUT_DIR", "./finetuned_model")
    max_new_tokens = int(os.getenv("FINETUNE_MAX_NEW_TOKENS", "128"))

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)

    inputs = tokenizer(question, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune and run a local LLM")
    parser.add_argument("mode", choices=["train", "ask"], help="train model or ask a question")
    parser.add_argument("--question", type=str, default="", help="question for ask mode")
    args = parser.parse_args()

    if args.mode == "train":
        train_finetuned_model()
        return

    question = args.question.strip() or input("Enter question: ").strip()
    if not question:
        raise ValueError("Question is required in ask mode.")

    print(answer_with_finetuned_model(question))


if __name__ == "__main__":
    main()
