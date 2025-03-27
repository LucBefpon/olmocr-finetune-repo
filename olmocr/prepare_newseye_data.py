import json
import base64
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoProcessor
from olmocr.prompts import build_finetuning_prompt

def make_response_json(text):
    """
    Create a JSON string with the expected fields for fine-tuning.
    """
    response = {
        "primary_language": "de",
        "is_rotation_valid": True,
        "rotation_correction": 0,
        "is_table": False,
        "is_diagram": False,
        "natural_text": text
    }
    return json.dumps(response, ensure_ascii=False)

def process_sample(example):
    """
    Process one sample: construct the training prompt, tokenize the input and the ground truth,
    and generate the final training example with input_ids, attention_mask, labels, and image features.
    """
    # Extract the ground truth text and image (as a PIL image)
    text = example["text"]
    image = example["image"]

    # Build the prompt using an empty anchor text (suitable for line-level OCR)
    prompt = build_finetuning_prompt("")

    # Create a chat message with the prompt and a base64-encoded image.
    # (This follows the format expected by olmOCR’s processor.)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": base64.b64encode(image.tobytes()).decode('utf-8')}
            ]
        }
    ]

    # Initialize the processor (this loads both the tokenizer and image processor).
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True)
    
    # Generate the complete prompt text using the chat template.
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenize the input (which includes the prompt and embedded image data)
    inputs = processor(text=[chat_text], images=[image], return_tensors="pt")
    input_ids = inputs["input_ids"][0].numpy()           # shape: (prompt_length,)
    attention_mask = inputs["attention_mask"][0].numpy()

    # Get image features – these will be used by the model.
    pixel_values = inputs["pixel_values"][0].numpy()
    image_grid_thw = inputs.get("image_grid_thw")
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw[0].numpy()
    else:
        image_grid_thw = np.array([])

    # Prepare the target response as a JSON string and tokenize it.
    resp_str = make_response_json(text)
    resp_tokens = processor.tokenizer(resp_str, add_special_tokens=False)["input_ids"]
    # Append the model’s end-of-response token (this might be required by Qwen’s chat format)
    end_tokens = processor.tokenizer("<|im_end|>\n", add_special_tokens=False)["input_ids"]
    resp_tokens += end_tokens
    resp_tokens = np.array(resp_tokens)

    # Concatenate the prompt tokens and the response tokens to create the full sequence.
    full_input_ids = np.concatenate([input_ids, resp_tokens])
    full_attention_mask = np.ones_like(full_input_ids)

    # Create labels: mask out the prompt tokens with -100 so that loss is computed only on the response.
    labels = np.full_like(full_input_ids, fill_value=-100)
    labels[len(input_ids):] = resp_tokens

    return {
        "input_ids": full_input_ids,
        "attention_mask": full_attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw
    }

def main():
    # Download the dataset from Hugging Face
    dataset = load_dataset("Teklia/NewsEye-Austrian-line")
    
    # Shuffle and split the dataset (90% train, 10% validation)
    dataset = dataset["train"].shuffle(seed=42)
    split_ds = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = split_ds["train"]
    val_ds = split_ds["test"]

    # Process the training set
    print("Processing training set...")
    train_processed = train_ds.map(process_sample, remove_columns=train_ds.column_names)
    
    # Process the validation set
    print("Processing validation set...")
    val_processed = val_ds.map(process_sample, remove_columns=val_ds.column_names)

    # Save the processed datasets to disk so they can be later loaded by the training script.
    print("Saving processed datasets to disk...")
    train_processed.save_to_disk("newseye_train_processed")
    val_processed.save_to_disk("newseye_val_processed")
    print("Data preparation completed.")

if __name__ == "__main__":
    main()
