import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import torch.nn as nn
import json
import random
import numpy as np
from unsloth import FastLanguageModel
from sentence_transformers import SentenceTransformer
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
from typing import List, Dict
import argparse

MAX_SEQ_LENGTH = 64000
MODEL_PATH = "sft/5k_variable"
CLASSIFIER_PATH = "best_model_robust_1.pth"
EMBEDDING_MODEL_NAME = "google/embeddinggemma-300m"
DEVICE = "cuda:2"

CLASSES = ['goodware', 'shipup', 'virlock', 'vobfus']


class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=[512, 256]):
        super(Classifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def json_validity_reward(completions: List[str], **kwargs) -> List[float]:
    rewards = []
    for completion in completions:
        try:
            json.loads(completion)
            rewards.append(1.0)
        except:
            rewards.append(-1.0)
    return rewards


def structure_reward(completions: List[str], **kwargs) -> List[float]:
    required_keys = {'files', 'read_files', 'write_files', 'delete_files', 
                     'keys', 'read_keys', 'write_keys', 'delete_keys', 
                     'executed_commands'}
    
    rewards = []
    for completion in completions:
        try:
            if completion in ['[]', '{}', '']:
                rewards.append(-10.0)
                continue
            data = json.loads(completion)
            if len(data) == 1 and 'summary' in data:
                if not isinstance(data['summary'], dict):
                    rewards.append(-5.0)
                    continue
                if set(data['summary'].keys()) == required_keys:
                    rewards.append(1.0)
                else:
                    rewards.append(-5.0)
            else:
                rewards.append(-5.0)
        except:
            rewards.append(-5.0)
    return rewards
        


def content_preservation_reward(completions: List[str], prompts: List[str], **kwargs) -> List[float]:
    rewards = []
    for completion, prompt in zip(completions, prompts):
        try:
            input_section = prompt.split("### Input:")[-1].split("### Response:")[0].strip()
            
            input_data = json.loads(input_section)
            output_data = json.loads(completion)
            
            if 'summary' not in input_data or 'summary' not in output_data:
                rewards.append(-5.0)
                continue
            
            input_summary = input_data['summary']
            output_summary = output_data['summary']
            
            if set(input_summary.keys()) != set(output_summary.keys()):
                rewards.append(-5.0)
                continue
            
            for key in input_summary.keys():
                for value in input_summary[key]:
                    if value not in output_summary[key]:
                        rewards.append(-5.0)
                        wrong = True
                        break
                if wrong:
                    break
            if not wrong:
                rewards.append(2.0)
            else:
                rewards.append(-5.0)
        except:
            rewards.append(-5.0)
    return rewards


def evasion_reward(completions: List[str], 
                   embedding_model: SentenceTransformer,
                   classifier: Classifier,
                   device: str,
                   **kwargs) -> List[float]:
    rewards = []
    
    with torch.no_grad():
        for completion in completions:
            try:
                data = json.loads(completion)
                
                text = str(data)
                
                embedding = embedding_model.encode([text], show_progress_bar=False)
                embedding_tensor = torch.FloatTensor(embedding).to(device)
                
                logits = classifier(embedding_tensor)
                predicted_class_idx = torch.argmax(logits, dim=1).item()
                predicted_class = CLASSES[predicted_class_idx]
                
                if predicted_class == 'goodware':
                    rewards.append(1.0)
                else:
                    rewards.append(-1.0)
            except:
                rewards.append(-1.0)
    
    return rewards


def combined_reward_function(
    prompts: List[str],
    completions: List[str],
    embedding_model: SentenceTransformer,
    classifier: Classifier,
    device: str,
    **kwargs
) -> List[float]:
    completions = [completion.split("\n")[-1].split("### Response:\n")[-1].replace("<|end_of_text|>", "").replace("<|im_end|>", "\n")
         for completion in completions]

    json_rewards = json_validity_reward(completions)
    struct_rewards = structure_reward(completions)
    content_rewards = content_preservation_reward(completions, prompts)
    evasion_rewards = evasion_reward(completions, embedding_model, classifier, device)
    
    weights = {
        'json': 2.0,
        'structure': 4.0,
        'evasion': 5.0,
        'content': 2.0
    }
    
    combined_rewards = []
    
    for j, s, e, c in zip(json_rewards, struct_rewards, evasion_rewards, content_rewards):
        reward = (weights['json'] * j + 
                    weights['structure'] * s + 
                    weights['evasion'] * e + 
                    weights['content'] * c)
        if j < 0 or s < 0 or c < 0:
            reward -= 5.0 
            reward -= (weights['evasion'] * e)
        
        combined_rewards.append(reward)
    
    with open("grpo2/rewards.txt", "a") as f:
        for j, s, e, c, reward, completion in zip(json_rewards, struct_rewards, evasion_rewards, content_rewards, combined_rewards, completions):
            f.write(f'JSON: {j}, Structure: {s}, Evasion: {e}, Content: {c}, Reward: {reward}\n')
            f.write(completion)
            f.write('_'*100)
            f.write('\n')
    return combined_rewards


def load_malware_data():
    malware_dirs = [
        'data/final_dataset/malware_dataset/shipup/reports_summary',
        'data/final_dataset/malware_dataset/virlock/reports_summary',
        'data/final_dataset/malware_dataset/vobfus/reports_summary'
    ]
    
    reports = []
    for directory in malware_dirs:
        if not os.path.exists(directory):
            continue
        print(f"Loading reports from {directory}")
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                path = os.path.join(directory, filename)
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                        reports.append(data)
                except:
                    continue
    
    return reports


def create_grpo_dataset(num_samples=1000):
    malware_reports = load_malware_data()
    
    if not malware_reports:
        raise ValueError("No malware data found!")
    
    print(f"Found {len(malware_reports)} malware samples. Generating {num_samples} prompts...")
    
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""
    
    instruction = "Generate an adversarial perturbation for the following malware log to bypass detection, while maintaining its malicious functionality. Output as valid JSON with the same structure."
    
    prompts = []
    for _ in range(num_samples):
        report = random.choice(malware_reports)
        input_json = json.dumps(report)
        prompt = alpaca_prompt.format(instruction, input_json, "")
        prompts.append(prompt)
    
    dataset = Dataset.from_dict({"prompt": prompts})
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="grpo2/")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=3)
    args = parser.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=MAX_SEQ_LENGTH,
    )
    
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    input_dim = 1024
    num_classes = len(CLASSES)
    classifier = Classifier(input_dim, num_classes).to(DEVICE)
    classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
    classifier.eval()
    
    dataset = create_grpo_dataset(num_samples=args.num_samples)
    
    def reward_fn(prompts, completions, **kwargs):
        return combined_reward_function(
            prompts=prompts,
            completions=completions,
            embedding_model=embedding_model,
            classifier=classifier,
            device=DEVICE,
            **kwargs
        )
    
    grpo_config = GRPOConfig(
        output_dir="grpo2/" + args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
    )
    
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        tokenizer=tokenizer,
        train_dataset=dataset,
        reward_funcs=reward_fn,
    )
    
    print("Starting GRPO training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    print("Saving final model...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("Done")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()