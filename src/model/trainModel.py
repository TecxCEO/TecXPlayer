"""
To prevent losing your progress or saving a "worse" version of the model due to overfitting, you should use a Checkpoint System. This script saves the model weights to a file only when the Validation Loss hits a new record low.
1. The Checkpoint Logic
Add this logic inside your training loop, specifically right after the Validation Phase:

"""

import tpsm as tm
from encoding_decoding import EncodeDecode as ed
##
if __name__ == "__main__":
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'stoi': ed.createTokens() # Saving the vocabulary is critical!
        #'stoi': stoi # Saving the vocabulary is critical!
    }
    device = 'gpu' if isavailable() else 'cpu'
    # Setup Example
    # vocab_size = 80  # Size of your 'stoi' map
    vocab_size =  len(checkpoint[ 'stoi']) # Size of your 'stoi' map
    max_epoch = 11
    model = tm.DictionaryTransformer(vocab_size=vocab_size, d_model=128, nhead=8, num_layers=4, num_classes=3)
    # model = DictionaryTransformer(vocab_size=vocab_size, d_model=128, nhead=20, num_layers=12, num_classes=3)
    input = imd.ImportDataset() #
    model(input) #
    best_val_loss = float('inf') # Start with infinity
    ##
    model.to(device)
    model.eval()
    for epoch in range(max_epoch):
        # Inside your epoch loop, after calculating avg_val_loss:
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save the model state
            torch.save(checkpoint, 'models/best_dictionary_model.pth')
            print(f"--> Saved new best model with Val Loss: {best_val_loss:.4f}")
        # Save progress
        torch.save(model.state_dict(), f"models/checkpoint_epoch_{epoch}.pth")
