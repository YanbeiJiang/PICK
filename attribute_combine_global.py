import os
import re
import json
import argparse
from tqdm import tqdm
from PIL import Image
from google import genai
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def save_json_data(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

def extract_emotion_confidence(response):
    try:
        pattern = r"Emotion:\s*(positive|negative)\s*;\s*Confidence:\s*(\d*\.?\d+)"
        match = re.search(pattern, response, re.IGNORECASE)
        emotion = match.group(1).lower()
        confidence = float(match.group(2))
        return emotion, confidence
    except Exception:
        return "positive", 0.50

def generate_emotion_by_description(image_path, client):
    prompt = (
        "As an emotional psychologist, analyze the image carefully and focus on the overall visual characteristics. "
        "Determine the underlying emotion is Negative (sadness, anger, etc.) or Positive (happiness, joy, etc.), "
        "and assign a confidence score (a float from 0 to 1, where 0 means no confidence and 1 means full confidence) "
        "indicating certainty in the emotional interpretation. "
        "Follow this exact output format and the emotion must only be 'postive' or 'negative': Emotion: xxx; Confidence: x.xx"
    )
    image = Image.open(image_path)

    attempts = 0
    while attempts < 5:
        try:
            response = client.models.generate_content(model="gemini-2.0-flash", contents=[prompt, image]).text
            break
        except Exception as e:
            attempts += 1
            print(f"Error occurred during generation (Attempt {attempts}/5): {e}")
            if attempts >= 5:
                print("Failed after 5 attempts. Exiting loop.")
                response = "Emotion: positive; Confidence: 0.50"
    return extract_emotion_confidence(response)

def main(data_folder, output_path):
    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    filenames = []
    labels = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".png"):
            filenames.append(os.path.join(data_folder, filename))
            if filename.startswith('0.'):
                labels.append("positive")
            elif filename.startswith('1.'):
                labels.append("negative")

    results = []
    valid_true = []
    valid_pred = []
    total = len(filenames)

    for image_path, label in tqdm(zip(filenames, labels), total=total):
        valid_true.append(label)
        description = ""
        emotion, confidence = generate_emotion_by_description(image_path, gemini_client)

        prediction = emotion if emotion in ["positive", "negative"] else "positive"
        valid_pred.append(prediction)

        results.append({
            "filename": image_path,
            "label": label,
            "prediction": prediction,
            "confidence": confidence,
            "description": description
        })

    summary = {
        "total_images": total,
        "accuracy_positive": accuracy_score(valid_true, valid_pred),
        "precision_positive": precision_score(valid_true, valid_pred, zero_division=0, pos_label='positive'),
        "recall_positive": recall_score(valid_true, valid_pred, zero_division=0, pos_label='positive'),
        "f1_positive": f1_score(valid_true, valid_pred, pos_label='positive'),
        "accuracy_negative": accuracy_score(valid_true, valid_pred),
        "precision_negative": precision_score(valid_true, valid_pred, zero_division=0, pos_label='negative'),
        "recall_negative": recall_score(valid_true, valid_pred, zero_division=0, pos_label='negative'),
        "f1_negative": f1_score(valid_true, valid_pred, pos_label='negative'),
    }

    print(f"Accuracy: {summary['accuracy_positive']:.4f}")
    print(f"Precision: {summary['precision_positive']:.4f}")
    print(f"Recall: {summary['recall_positive']:.4f}")
    print(f"F1 Score: {summary['f1_positive']:.4f}")

    save_json_data({"global_results": results, "summary": summary}, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default="./data/image_resized")
    parser.add_argument('--output_path', type=str, default="emotional_classification_results_global.json")
    args = parser.parse_args()

    main(args.data_folder, args.output_path)
