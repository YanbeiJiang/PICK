import argparse
import json
import os
import re
import numpy as np
from tqdm import tqdm
from google import genai
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

gemini_client = genai.Client(api_key="YOUR_API_KEY")

def load_json_data(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def save_json_data(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

def generate_prompt_by_attribute(objects):
    all_objects = objects[0]
    return f"As an emotional psychologist, analyze the relationship between the {all_objects} depicted within the green bounding box in the sketch drawing. Provide a concise description of how these objects relate to each other, and should not involve any emotional understanding. The output structure must be exactly the following: Description: xxx"

def generate_description_with_gemini(image_path):
    image = Image.open(image_path)
    attempts = 0
    while attempts < 5:
        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[generate_prompt_by_attribute(image_path.split("nearby_")[1].rsplit(".", 1)[0].split("_")), image]
            ).text
            print(response)
            break
        except Exception as e:
            attempts += 1
            print(f"Error during generation (Attempt {attempts}/5): {e}")
            if attempts >= 5:
                print("Failed after 5 attempts.")
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

def generate_emotion_by_description(image_path, text):
    prompt = (
        "As an emotional psychologist, carefully analyze the image by focusing on the interactions between objects in the scene, "
        "ensuring no details are overlooked in making the diagnosis. Determine the underlying emotion is Negative (sadness, anger, etc.) "
        "or Positive (happiness, joy, etc.), and assign a confidence score (a float from 0 to 1, where 0 means no confidence and 1 means full confidence) "
        "indicating certainty in the emotional interpretation. Follow this exact output format and the emotion must only be 'postive' or 'negative': "
        "Emotion: xxx; Confidence: x.xx"
    )
    image = Image.open(image_path)
    attempts = 0
    while attempts < 5:
        try:
            response = gemini_client.models.generate_content(model="gemini-2.0-flash", contents=[prompt, image]).text
            print(response)
            break
        except Exception as e:
            attempts += 1
            print(f"Error during generation (Attempt {attempts}/5): {e}")
            if attempts >= 5:
                print("Failed after 5 attempts.")
    return extract_emotion_confidence(response)

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

def main(json_path, output_path, use_description=False):
    full_data = load_json_data(json_path)
    data = dict(list(full_data.items()))

    print("Step 1: Generating descriptions...")
    data_with_description = {}
    for image_path in tqdm(data.keys()):
        if use_description:
            description = generate_description_with_gemini(image_path).split("Description: ")[1]
        else:
            description = ""
        data_with_description[image_path] = description

    print("Step 2: Predicting emotions...")
    prefix_distributions = {}
    for image_path, description in tqdm(data_with_description.items()):
        prefix = get_image_prefix(image_path)
        if prefix not in prefix_distributions:
            prefix_distributions[prefix] = {
                "pos_sum": 0,
                "neg_sum": 0,
                "count": 0,
                "paths": []
            }
        emotion, confidence = generate_emotion_by_description(image_path, description)
        if emotion == "positive":
            prefix_distributions[prefix]["pos_sum"] += confidence
        elif emotion == "negative":
            prefix_distributions[prefix]["neg_sum"] += confidence
        prefix_distributions[prefix]["count"] += 1
        prefix_distributions[prefix]["paths"].append(image_path)

    print("Step 3: Aggregating results...")
    for prefix, info in prefix_distributions.items():
        avg_pos = info["pos_sum"] / info["count"]
        avg_neg = info["neg_sum"] / info["count"]
        predicted_label = "positive" if avg_pos >= avg_neg else "negative"
        prefix_distributions[prefix]["final_distribution"] = {"pos": avg_pos, "neg": avg_neg}
        prefix_distributions[prefix]["predicted_label"] = predicted_label

    valid_true = []
    valid_pred = []
    for prefix, info in prefix_distributions.items():
        if not info["paths"]:
            continue
        ground_truth = get_ground_truth(info["paths"][0])
        predicted = info["predicted_label"]
        valid_true.append(ground_truth)
        valid_pred.append(predicted)

    accuracy_0 = accuracy_score(valid_true, valid_pred)
    precision_0 = precision_score(valid_true, valid_pred, zero_division=0, pos_label='positive')
    recall_0 = recall_score(valid_true, valid_pred, zero_division=0, pos_label='positive')
    f1_0 = f1_score(valid_true, valid_pred, pos_label='positive')
    accuracy_1 = accuracy_score(valid_true, valid_pred)
    precision_1 = precision_score(valid_true, valid_pred, zero_division=0, pos_label='negative')
    recall_1 = recall_score(valid_true, valid_pred, zero_division=0, pos_label='negative')
    f1_1 = f1_score(valid_true, valid_pred, pos_label='negative')

    summary = {
        "total_images": len(valid_true),
        "accuracy_positive": accuracy_0,
        "precision_positive": precision_0,
        "recall_positive": recall_0,
        "f1_positive": f1_0,
        "accuracy_negative": accuracy_1,
        "precision_negative": precision_1,
        "recall_negative": recall_1,
        "f1_negative": f1_1,
    }

    print(f"Accuracy: {accuracy_0:.4f}")
    print(f"Precision: {precision_0:.4f}")
    print(f"Recall: {recall_0:.4f}")
    print(f"F1 Score: {f1_0:.4f}")

    results = {
        "instance_level": data_with_description,
        "prefix_level": prefix_distributions,
        "summary": summary
    }

    save_json_data(results, output_path)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="./data/image_bbox/extracted_features_object.json")
    parser.add_argument("--output_path", type=str, default="emotional_classification_results_relation.json")
    parser.add_argument("--use_description", action="store_true", help="Whether to generate descriptions before emotion classification")
    args = parser.parse_args()

    main(args.json_path, args.output_path, args.use_description)
