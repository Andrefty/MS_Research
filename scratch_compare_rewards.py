import json
import glob
import numpy as np
import sys
import os

sys.path.insert(0, '/export/home/acs/stud/t/tudor.farcasanu/SSL_research')
from utils.response_parser import parse_model_response as parse_response

rollout_files = glob.glob('/export/home/acs/stud/t/tudor.farcasanu/SSL_research/checkpoints/grpo_qwen3_4b_verl/train_rollout/*.jsonl')
rollout_files = sorted(rollout_files, key=os.path.getmtime, reverse=True)[:5]

old_rewards_vuln = []
new_rewards_vuln = []
old_rewards_patched = []
new_rewards_patched = []

for fpath in rollout_files:
    with open(fpath, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            
            output = data.get('output', '')
            gts_str = data.get('gts', '{}')
            old_rew = data.get('score', 0.0)
            
            try:
                gt = json.loads(gts_str) if isinstance(gts_str, str) else gts_str
            except:
                continue
                
            is_vuln = gt.get('is_vulnerable', False)
            gt_lines = gt.get('ground_truth_lines', [])
            
            result = parse_response(output)
            new_rew = 0.00
            
            if result.status in ['VALID', 'INVALID_CLASSIFICATION']:
                pred_vuln = (result.classification == 'VULNERABLE')
                pred_lines = result.important_lines
                    
                if result.classification is None:
                    new_rew = 0.15
                elif pred_vuln == is_vuln:
                    new_rew = 0.75
                    
                    if len(gt_lines) == 0:
                        if len(pred_lines) == 0:
                            new_rew = 1.25
                    else:
                        correct_lines = len(set(pred_lines) & set(gt_lines))
                        pct = correct_lines / len(gt_lines)
                        
                        capped_pct = min(1.0, pct / 0.75)
                        new_rew = 0.75 + (0.50 * capped_pct)
                else:
                    new_rew = 0.15
            else:
                new_rew = 0.00
            
            if is_vuln:
                old_rewards_vuln.append(old_rew)
                new_rewards_vuln.append(new_rew)
            else:
                old_rewards_patched.append(old_rew)
                new_rewards_patched.append(new_rew)

print('--- VULNERABLE SAMPLES ---')
if old_rewards_vuln:
    print(f'Count: {len(old_rewards_vuln)}')
    print(f'Mean Old: {np.mean(old_rewards_vuln):.3f}')
    print(f'Mean New: {np.mean(new_rewards_vuln):.3f}')
    print(f'Std Old:  {np.std(old_rewards_vuln):.3f}')
    print(f'Std New:  {np.std(new_rewards_vuln):.3f}')

print('\n--- PATCHED SAMPLES ---')
if old_rewards_patched:
    print(f'Count: {len(old_rewards_patched)}')
    print(f'Mean Old: {np.mean(old_rewards_patched):.3f}')
    print(f'Mean New: {np.mean(new_rewards_patched):.3f}')
    print(f'Std Old:  {np.std(old_rewards_patched):.3f}')
    print(f'Std New:  {np.std(new_rewards_patched):.3f}')
    
print('\n--- ALL SAMPLES ---')
if old_rewards_vuln or old_rewards_patched:
    all_old = old_rewards_vuln + old_rewards_patched
    all_new = new_rewards_vuln + new_rewards_patched
    print(f'Mean Old: {np.mean(all_old):.3f}')
    print(f'Mean New: {np.mean(all_new):.3f}')
    print(f'Std Old:  {np.std(all_old):.3f}')
    print(f'Std New:  {np.std(all_new):.3f}')
