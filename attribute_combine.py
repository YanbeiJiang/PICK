import argparse
import json
import os
import re
import numpy as np
from tqdm import tqdm
from google import genai
from openai import OpenAI
from PIL import Image
import time
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set seed
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Emotion classification pipeline")

    parser.add_argument("--json_path", type=str, default="./data/image_bbox/extracted_features_object.json")
    parser.add_argument("--csv_path", type=str, default="knowledge_base_with_embeddings_soft_label_cleaned.csv")
    parser.add_argument("--output_path", type=str, default="emotional_classification_results_object.json")
    parser.add_argument("--gemini_key", type=str, default="YOUR_KEY_HERE")
    parser.add_argument("--gpt_key", type=str, default="YOUR_KEY_HERE")
    parser.add_argument("--vlm_model", type=str, default="HuggingFaceM4/Idefics3-8B-Llama3")
    parser.add_argument("--valid_words", type=str, nargs="+", default=["house", "tree", "person"])

    return parser.parse_args()

args = parse_args()

# Configure API clients
gemini_client = genai.Client(api_key=args.gemini_key)
gpt_client = OpenAI(api_key=args.gpt_key)

# Load CSV
csv_data = pd.read_csv(args.csv_path)
csv_data["head_embedding"] = csv_data["head_embedding"].apply(lambda x: np.array(eval(x)) if isinstance(x, str) else x)

# Load processor/model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
caption_processor = AutoProcessor.from_pretrained(args.vlm_model)
caption_model = AutoModelForVision2Seq.from_pretrained(
    args.vlm_model, torch_dtype=torch.bfloat16, _attn_implementation="flash_attention_2"
).to(device)

def load_json_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def save_json_data(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

def generate_prompt_by_attribute(objects, attribute):
    all_objects = ",".join(list(set(objects)))
    return f"Acting as a emotional psychologist, provide a concise and complete sentence description of the {all_objects} depicted within the green bounding box in the sketch drawing. Think carefully and the sentence should focus on the following: {attribute}, and should not involve any emotional words. The output structure must be exactly the following: Description: xxx"

def generate_description(image_path, attribute):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": generate_prompt_by_attribute(image_path.split("nearby_")[1].split(".png")[0].split("_"), attribute)},
            ],
        }
    ]
    image = load_image(image_path)
    prompt = caption_processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = caption_processor(text=prompt, images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    generated_ids = caption_model.generate(**inputs, max_new_tokens=500)
    response = caption_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].split("Assistant: ")[1]
    print(response)
    return response

def extract_emotion_confidence(response):
    try:
        pattern = r"Emotion:\s*(positive|negative)\s*;\s*Confidence:\s*(\d*\.?\d+)"
        match = re.search(pattern, response, re.IGNORECASE)
        emotion = match.group(1).lower()
        confidence = float(match.group(2))
        return emotion, confidence
    except Exception:
        return "positive", 0.50

def get_embedding(text, model="text-embedding-3-large"):
    response = gpt_client.embeddings.create(input=text, model=model)
    return response.data[0].embedding

def extract_emotion_kb(description):
    description_embedding = get_embedding(description) 
    similarities = cosine_similarity([description_embedding], np.vstack(csv_data["head_embedding"]))
    top_index = similarities[0].argsort()[-1]
    similarity = similarities[0][top_index]
    label = json.loads(csv_data.iloc[top_index]["extracted_label"])
    positive_score = float(label["Positive"].strip('%'))
    negative_score = float(label["Negative"].strip('%'))
    total = positive_score + negative_score
    return {"pos": positive_score / total, "neg": negative_score / total}, similarity

def generate_emotion_by_description(attribute, text, image_path):
    prompt = f"As an emotional psychologist, analyze the following: 1. the image, 2. this attribute about the object in the bounding box: {attribute} 3. this description based on the image and attribute: {text}, ensuring no details are overlooked in making the diagnosis. Think carefully and determine the underlying emotion is Negative (sadness, anger, etc.) or Positive (happiness, joy, etc.), and assign a confidence score (a float from 0 to 1). Follow this exact output format and the emotion must only be 'positive' or 'negative': Emotion: xxx; Confidence: x.xx"
    image = Image.open(image_path)
    attempts = 0
    response = "Emotion: positive; Confidence: 0.50"
    while attempts < 5:
        try:
            response = gemini_client.models.generate_content(model="gemini-2.0-flash", contents=[prompt, image]).text
            print(response)
            break
        except Exception as e:
            attempts += 1
            print(f"Error occurred during generation (Attempt {attempts}/5): {e}")
            time.sleep(10)
    return response

def get_image_prefix(image_path):
    filename = os.path.basename(image_path)
    prefix = re.match(r'(\d+\.\d+)', filename)
    return prefix.group(1) if prefix else None

def get_ground_truth(image_path):
    filename = os.path.basename(image_path)
    if filename.startswith('0.'):
        return "positive"
    elif filename.startswith('1.'):
        return "negative"
    return None

def combine_emotions(emotion_llm_list, emotion_distributions_kb_list, confidences_list, similarities_list):
    assert len(emotion_llm_list) == len(emotion_distributions_kb_list) == len(confidences_list) == len(similarities_list)
    combined_pos_scores = []
    combined_neg_scores = []
    for i in range(len(emotion_llm_list)):
        llm_emotion = emotion_llm_list[i]
        llm_confidence = confidences_list[i]
        kb_dist = emotion_distributions_kb_list[i]
        kb_similarity = similarities_list[i]
        llm_dist = {"pos": 1.0 if llm_emotion == "positive" else 0.0, "neg": 1.0 if llm_emotion == "negative" else 0.0}
        exp_llm = np.exp(llm_confidence)
        exp_kb = np.exp(kb_similarity)
        total_exp = exp_llm + exp_kb
        llm_weight = exp_llm / total_exp
        kb_weight = exp_kb / total_exp
        combined_pos = (llm_weight * llm_dist["pos"]) + (kb_weight * kb_dist["pos"])
        combined_neg = (llm_weight * llm_dist["neg"]) + (kb_weight * kb_dist["neg"])
        total = combined_pos + combined_neg
        combined_pos /= total
        combined_neg /= total
        combined_pos_scores.append(combined_pos)
        combined_neg_scores.append(combined_neg)
    avg_pos = np.mean(combined_pos_scores)
    avg_neg = np.mean(combined_neg_scores)
    total = avg_pos + avg_neg
    return avg_pos / total, avg_neg / total

def select_instance(json_data, valid_words):
    selected_instances = {}
    for filepath, attributes in json_data.items():
        first_word = os.path.basename(filepath).split("_")[1]
        if first_word in valid_words:
            selected_instances[filepath] = attributes
    return selected_instances

def main():
    full_data = load_json_data(args.json_path)
    data = select_instance(full_data, set(args.valid_words))
    data = dict(list(data.items()))
    print("Step 1: Generating descriptions...")
    for image_path, attributes in tqdm(data.items()):
        data[image_path] = {
            "attributes": attributes,
            "descriptions": []
        }
        for attribute in attributes:
            description = generate_description(image_path, attribute)
            if "Description: " in description:
                description = description.split("Description: ")[1]
            data[image_path]["descriptions"].append(description)
    print("Step 2: Getting emotion distributions...")
    for image_path, info in tqdm(data.items()):
        emotion_llm, emotion_distributions_kb, confidences, similarities = [], [], [], []
        for i, description in enumerate(info["descriptions"]):
            emotion_conf_raw = generate_emotion_by_description(info["attributes"][i], description, image_path)
            emotion, confidence = extract_emotion_confidence(emotion_conf_raw)
            emotion_llm.append(emotion)
            distribution_kb, similarity = extract_emotion_kb(description)
            emotion_distributions_kb.append(distribution_kb)
            confidences.append(confidence)
            similarities.append(similarity)
        data[image_path]["emotion_llm"] = emotion_llm
        data[image_path]["emotion_distributions_kb"] = emotion_distributions_kb
        data[image_path]["confidences"] = confidences
        data[image_path]["similarities"] = similarities
    print("Step 3: Averaging distributions for each instance...")
    for image_path, info in data.items():
        avg_pos, avg_neg = combine_emotions(info["emotion_llm"], info["emotion_distributions_kb"], info["confidences"], info["similarities"])
        data[image_path]["instance_distribution"] = {"pos": avg_pos, "neg": avg_neg}
    print("Step 4: Averaging over instances with the same prefix...")
    prefix_distributions = {}
    for image_path, info in data.items():
        prefix = get_image_prefix(image_path)
        if prefix not in prefix_distributions:
            prefix_distributions[prefix] = {"pos_sum": 0, "neg_sum": 0, "count": 0, "paths": []}
        prefix_distributions[prefix]["pos_sum"] += info["instance_distribution"]["pos"]
        prefix_distributions[prefix]["neg_sum"] += info["instance_distribution"]["neg"]
        prefix_distributions[prefix]["count"] += 1
        prefix_distributions[prefix]["paths"].append(image_path)
    for prefix, info in prefix_distributions.items():
        avg_pos = info["pos_sum"] / info["count"]
        avg_neg = info["neg_sum"] / info["count"]
        predicted_label = "positive" if avg_pos >= avg_neg else "negative"
        prefix_distributions[prefix]["final_distribution"] = {"pos": avg_pos, "neg": avg_neg}
        prefix_distributions[prefix]["predicted_label"] = predicted_label
    print("Step 5: Calculating accuracy...")
    valid_true, valid_pred = [], []
    for prefix, info in prefix_distributions.items():
        if not info["paths"]:
            continue
        ground_truth = get_ground_truth(info["paths"][0])
        predicted = info["predicted_label"]
        valid_true.append(ground_truth)
        valid_pred.append(predicted)
    summary = {
        "accuracy_positive": accuracy_score(valid_true, valid_pred),
        "precision_positive": precision_score(valid_true, valid_pred, zero_division=0, pos_label='positive'),
        "recall_positive": recall_score(valid_true, valid_pred, zero_division=0, pos_label='positive'),
        "f1_positive": f1_score(valid_true, valid_pred, pos_label='positive'),
        "accuracy_negative": accuracy_score(valid_true, valid_pred),
        "precision_negative": precision_score(valid_true, valid_pred, zero_division=0, pos_label='negative'),
        "recall_negative": recall_score(valid_true, valid_pred, zero_division=0, pos_label='negative'),
        "f1_negative": f1_score(valid_true, valid_pred, pos_label='negative'),
    }
    print(summary)
    results = {
        "instance_level": data,
        "prefix_level": prefix_distributions,
        "summary": summary
    }
    save_json_data(results, args.output_path)
    return results

if __name__ == "__main__":
    main()
