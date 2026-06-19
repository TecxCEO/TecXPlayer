import json
import torch
from torch.utils.data import Dataset, DataLoader

class VerifiedCubicPuzzleDataset(Dataset):
    def __init__(self, json_path, dataset_size=100):
        with open(json_path, "r") as f:
            data = json.load(f)
        
        flattened_context = []
        
        # We expect exactly 1 entry in the root array for our target sequence block
        sequence_data = data[0]
        batches = sequence_data["context_window_batches"]
        
        # 1. Validate Total Batches Length
        assert len(batches) == 18, f"Context Window Error: Expected exactly 18 batches, found {len(batches)}"
        
        for expected_batch_idx, batch in enumerate(batches):
            # 2. Strict Zero-Based Index Validation for Batches (0 to 17)
            current_batch_idx = batch["batch_index"]
            assert current_batch_idx == expected_batch_idx, \
                f"Index Error: Expected batch index {expected_batch_idx}, but found {current_batch_idx}"
                
            steps = batch["steps"]
            # 3. Validate Total Steps Length Per Batch
            assert len(steps) == 20, \
                f"Step Count Error: Batch {current_batch_idx} must have exactly 20 steps, found {len(steps)}"
            
            for expected_step_idx, step in enumerate(steps):
                # 4. Strict Zero-Based Index Validation for Steps (0 to 19)
                current_step_idx = step["step_index"]
                assert current_step_idx == expected_step_idx, \
                    f"Index Error: Batch {current_batch_idx}, expected step index {expected_step_idx}, but found {current_step_idx}"
                
                # Extract and reconstruct tokens in the precise sequence requested
                start_token = [step["start_token"]]
                fwd_bwd_token = [step["fwd_bwd_token"]]
                current_state = step["current_state"]
                move_token = [step["move_token"]]
                resulting_state = step["resulting_state"]
                eo_token = [step["eo_token"]]
                
                # Check individual state constraints (20 tokens each)
                assert len(current_state) == 20, f"Format Error: Current state in Batch {current_batch_idx}, Step {current_step_idx} must be 20 tokens."
                assert len(resulting_state) == 20, f"Format Error: Resulting state in Batch {current_batch_idx}, Step {current_step_idx} must be 20 tokens."
                
                # Assemble the 44-token flat block
                step_tokens = start_token + fwd_bwd_token + current_state + move_token + resulting_state + eo_token
                assert len(step_tokens) == 44, f"Alignment Error: Step tokens layout must be 44. Got {len(step_tokens)}"
                
                flattened_context.extend(step_tokens)
        
        # 5. Final Context Size Verification (18 * 20 * 44 = 15,840)
        assert len(flattened_context) == 15840, f"Total Context Error: Expected 15840 tokens, generated {len(flattened_context)}"
        
        # Duplicate to fill the dataset allocation for stable overtraining loops
        self.sequences = [torch.tensor(flattened_context, dtype=torch.long) for _ in range(dataset_size)]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = self.sequences[idx][:-1]
        y = self.sequences[idx][1:]
        return x, y

# ==========================================
# RUN VALIDATION CHECK
# ==========================================
if __name__ == "__main__":
    try:
        # Load and validate your JSON file
        dataset = VerifiedCubicPuzzleDataset("training_data.json", dataset_size=10)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        for batch_x, batch_y in dataloader:
            print("\n" + "="*50)
            print("SUCCESS: JSON Zero-Based Indexing is 100% Valid!")
            print(f"Verified Input Vector Shape:  {batch_x.shape} (Context: 15839 targets)")
            print(f"Verified Target Vector Shape: {batch_y.shape} (Shifted by 1)")
            print("="*50 + "\n")
            break
    except AssertionError as error:
        print(f"\n❌ VALIDATION FAILED: {error}\n")
    except FileNotFoundError:
        print("\nDataset script compiled. Create your 'training_data.json' starting with indices 0 to test.\n")

