import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import ast
from tqdm import tqdm
import time
import os
import argparse

# Set seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(model, tokenizer, epoch, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"reward_model_epoch_{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer,
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

class EmotionDataset(Dataset):
    def __init__(self, csv_file, tokenizer):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.heads = self.data['head'].values
        self.scores = self.data['extracted_label'].values

    def __len__(self):
        return len(self.heads)

    def __getitem__(self, idx):
        text = self.heads[idx]
        score = self.scores[idx]
        label_dict = ast.literal_eval(score)
        positive = float(label_dict["Positive"].replace('%', '')) / 100
        negative = float(label_dict["Negative"].replace('%', '')) / 100
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs, torch.tensor([positive, negative], dtype=torch.bfloat16)

class RewardModel(nn.Module):
    def __init__(self, model_name, device, tokenizer):
        super(RewardModel, self).__init__()
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.config.pad_token_id = tokenizer.pad_token_id

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return torch.log_softmax(outputs.logits, dim=1)

class BatchMetrics:
    def __init__(self):
        self.running_loss = 0.0
        self.count = 0
        self.start_time = time.time()
    
    def update(self, loss, batch_size):
        self.running_loss += loss * batch_size
        self.count += batch_size
    
    def get_average_loss(self):
        return self.running_loss / self.count if self.count > 0 else 0.0
    
    def get_time_elapsed(self):
        return time.time() - self.start_time
    
    def reset(self):
        self.running_loss = 0.0
        self.count = 0
        self.start_time = time.time()

def train_reward_model(model, train_loader, max_epochs, learning_rate, log_interval, tokenizer, checkpoint_dir):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    total_batches = len(train_loader)
    print(f"Total number of batches: {total_batches}")
    model.train()

    for i, epoch in enumerate(range(max_epochs)):
        metrics = BatchMetrics()
        progress_bar = tqdm(enumerate(train_loader), total=total_batches, 
                          desc=f"Epoch {epoch + 1}/{max_epochs}")
        
        for batch_idx, (inputs, labels) in progress_bar:
            batch_size = labels.size(0)
            input_ids = inputs['input_ids'].to(model.device)
            attention_mask = inputs['attention_mask'].to(model.device)
            labels = labels.to(model.device)

            optimizer.zero_grad()
            predicted_scores = model(input_ids, attention_mask)
            loss = criterion(predicted_scores, labels)
            loss.backward()
            optimizer.step()
            metrics.update(loss.item(), batch_size)
            
            if (batch_idx + 1) % log_interval == 0:
                avg_loss = metrics.get_average_loss()
                elapsed_time = metrics.get_time_elapsed()
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'time/batch': f'{elapsed_time/batch_idx:.3f}s',
                    'samples': metrics.count
                })
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Average Loss: {metrics.get_average_loss():.4f}")
        print(f"Total Samples Processed: {metrics.count}")
        print(f"Total Time: {metrics.get_time_elapsed():.2f}s")
        print("-" * 50)
        
        if ((i+1) % 10) == 0:
            save_checkpoint(model, tokenizer, epoch, checkpoint_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a reward model using soft emotion labels.")

    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Pretrained model name or path.")
    parser.add_argument('--csv_file', type=str, required=True, help="Path to CSV file with soft labels.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training.")
    parser.add_argument('--max_epochs', type=int, default=50, help="Number of training epochs.")
    parser.add_argument('--learning_rate', type=float, default=5e-5, help="Learning rate.")
    parser.add_argument('--log_interval', type=int, default=10, help="Log every n batches.")
    parser.add_argument('--checkpoint_dir', type=str, default="/data/gpfs/projects/punim1996/Data/HTP/checkpoints/reward_model", help="Directory to save model checkpoints.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")

    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )

    dataset = EmotionDataset(csv_file=args.csv_file, tokenizer=tokenizer)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    reward_model = RewardModel(args.model_name, device, tokenizer)
    train_reward_model(reward_model, train_loader, args.max_epochs, args.learning_rate, args.log_interval, tokenizer, args.checkpoint_dir)
