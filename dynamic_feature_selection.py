import argparse
import os
import re
import gc
import time
import json
import math
import torch
import base64
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import Counter
from datasets import Dataset
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForSequenceClassification,
    AutoModelForVision2Seq,
    Qwen2_5_VLForConditionalGeneration
)
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from google import genai
from google.genai.types import GenerateContentConfig
from openai import OpenAI
from qwen_vl_utils import process_vision_info
from transformers.image_utils import load_image


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def predict(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        log_probs = torch.log_softmax(outputs.logits, dim=1)
        probs = torch.exp(log_probs)
    return probs[0, 0].item(), probs[0, 1].item()

def generate_prompt_by_attribute(objects, attribute):
    all_objects = objects[0]
    return f"Acting as a emotional psychologist, provide a concise and complete sentence description of the {all_objects} depicted within the green bounding box in the sketch drawing. The sentence should focus on the {attribute} of the {all_objects}, and should not involve any emotional understanding. The output structure must be exactly the following: Description: xxx"

def highest_frequency_word(words):
    counts = Counter(words)
    max_freq = max(counts.values())
    for word in words:
        if counts[word] == max_freq:
            return word

def compute_reward(scores, eps=1e-8):
    scores = [max(score, eps) for score in scores]
    total = sum(scores)
    scores = [score / total for score in scores]
    entropy = -sum(p * math.log(p) for p in scores if p > 0)
    max_entropy = math.log(len(scores))
    return 1 - (entropy / max_entropy)

def reward_func(prompts, completions, **kwargs):
    filepath = prompts[0]["content"][0]["image"]
    rewards = []
    generated_answer_list = []
    for i, completion in enumerate(completions):
        generated_answer = completion.split("\nassistant\n")[1]
        generated_answer_list.append(generated_answer)
        image = Image.open(filepath)
        attempts = 0
        while attempts < 5:
            try:
                response = gemini_client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[generate_prompt_by_attribute(filepath.split("nearby_")[1].rsplit(".", 1)[0].split("_"), generated_answer), image],
                    config=GenerateContentConfig(seed=42, temperature=0)).text
                break
            except Exception as e:
                attempts += 1
                print(f"Error occurred (Attempt {attempts}/5): {e}")
                time.sleep(10)
                if attempts >= 5:
                    print("Failed after 5 attempts.")
        if "Description: " in response:
            response = response.split("Description: ")[1].strip()
        positive_score, negative_score = predict(reward_model, reward_tokenizer, response)
        final_score = compute_reward([positive_score, negative_score])
        print("final_score: ", final_score)
        rewards.append(final_score)
    new_feature_list = extract_feature_dict[filepath].copy()
    if len(new_feature_list) == NUM_FIX_FEATURES + k:
        new_feature_list.append(highest_frequency_word(generated_answer_list))
    elif len(new_feature_list) == NUM_FIX_FEATURES + k + 1:
        new_feature_list[-1] = highest_frequency_word(generated_answer_list)
    extract_feature_dict[filepath] = new_feature_list
    print("-------------------------")
    return rewards

def dataset_prepare(extract_feature_dict, data_folder):
    data = []
    for filename in tqdm(os.listdir(data_folder)):
        if filename.endswith(".png"):
            base = filename.rsplit('.', 1)[0]
            parts = base.split("nearby")
            if len(parts) > 1:
                object_str = parts[1].lstrip('_')
                object_name = object_str.split('_')[0]
            excluded_features = extract_feature_dict[os.path.join(data_folder, filename)]
            text = (f"Given the provided hand-drawn sketch, focus on the object '{object_name}' in the green bounding box. "
                    "Identify one detailed visual attribute of this object that contributes to understanding psychological positive or negative emotions. "
                    f"Avoid mentioning these attributes: {', '.join(excluded_features)} and Color. "
                    "Be specific and provide ONLY a short phrase.")
            messages = {"prompt": {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": os.path.join(data_folder, filename)},
                        {"type": "text", "text": text},
                    ],
                }}
            data.append(messages)
    return Dataset.from_list(data)

def load_checkpoint(checkpoint_path, model_name):
    checkpoint = torch.load(checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    )
    new_state_dict = {key.replace("model.model.", "model."): val for key, val in checkpoint['model_state_dict'].items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    tokenizer = checkpoint['tokenizer']
    return model, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default="./checkpoints/reward_model/reward_model_epoch_49.pt")
    parser.add_argument('--csv_path', type=str, default="knowledge_base_with_embeddings_soft_label_cleaned.csv")
    parser.add_argument('--data_folder', type=str, default="./data/image_bbox/level_3_resized")
    parser.add_argument('--extract_json_path', type=str, default="./data/image_bbox/extracted_features_object.json")
    parser.add_argument('--output_dir', type=str, default="./checkpoints/GRPO_model")
    parser.add_argument('--run_name', type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_DYNAMIC_FEATURES = 2
    NUM_FIX_FEATURES = 2

    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    gpt_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    csv_data = pd.read_csv(args.csv_path)
    csv_data["head_embedding"] = csv_data["head_embedding"].apply(lambda x: np.array(eval(x)) if isinstance(x, str) else x)

    # Initialize lists to store filenames and labels
    filenames = []
    data = {}
    for filename in os.listdir(args.data_folder):
        if filename.endswith(".png"):
            data[os.path.join(args.data_folder, filename)] = ["Position", "Size"]
    with open(args.extract_json_path, "w") as f:
        json.dump(data, f)

    reward_model, reward_tokenizer = load_checkpoint(args.checkpoint_path, "Qwen/Qwen2.5-7B-Instruct")

    for k in range(NUM_DYNAMIC_FEATURES):
        with open(args.extract_json_path, "r") as f:
            extract_feature_dict = json.load(f)

        dataset = dataset_prepare(extract_feature_dict, args.data_folder)

        training_args = GRPOConfig(
            output_dir=args.output_dir,
            run_name=args.run_name,
            learning_rate=5e-5,
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=0.1,
            warmup_ratio=0.1,
            lr_scheduler_type='cosine',
            logging_steps=1,
            bf16=True,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            num_generations=2,
            max_prompt_length=256,
            max_completion_length=786,
            num_train_epochs=1,
            save_steps=100,
            max_grad_norm=0.1,
            report_to="wandb",
            log_on_each_node=False,
            temperature=100
        )

        peft_config = LoraConfig(
            r=16,
            lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
        )

        update_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False
        ).to(device)

        update_tokenizer = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        update_tokenizer.tokenizer.padding_side = "left"

        trainer = GRPOTrainer(
            model=update_model,
            processing_class=update_tokenizer,
            reward_funcs=[reward_func],
            args=training_args,
            train_dataset=dataset,
            peft_config=peft_config
        )
        trainer.train()

        with open(args.extract_json_path, "w") as f:
            json.dump(extract_feature_dict, f)

        del update_model, trainer, update_tokenizer
        gc.collect()
