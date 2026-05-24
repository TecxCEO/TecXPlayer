"""
To prevent losing your progress or saving a "worse" version of the model due to overfitting, you should use a Checkpoint System. This script saves the model weights to a file only when the Validation Loss hits a new record low.
1. The Checkpoint Logic
Add this logic inside your training loop, specifically right after the Validation Phase:

"""
from encoding_decoding import EncodeDecode as ed
best_val_loss = float('inf') # Start with infinity

# Inside your epoch loop, after calculating avg_val_loss:
if avg_val_loss < best_val_loss:
    best_val_loss = avg_val_loss
    
    # Save the model state
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'stoi': ed.createTokens() # Saving the vocabulary is critical!
        #'stoi': stoi # Saving the vocabulary is critical!
    }
    
    torch.save(checkpoint, 'best_dictionary_model.pth')
    print(f"--> Saved new best model with Val Loss: {best_val_loss:.4f}")

"""
2. Why save the stoi?
If you restart your script later and your data has changed, a new stoi map might be generated. If the character indices don't match the ones used during training, the model will output complete nonsense. Saving it in the .pth file ensures they stay together.
3. How to Load the Model Later
When you are ready to use the model for real predictions (or to resume training), use this:

"""

# 1. Recreate the model architecture
model = DictionaryTransformer(...) 

# 2. Load the saved data
checkpoint = torch.load('best_dictionary_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
stoi = checkpoint['stoi'] # Recover your vocabulary

model.eval() # Set to evaluation mode
print("Model loaded and ready!")

"""
4. Pro-Tip: Early Stopping
If the validation loss doesn't improve for 10 epochs in a row, you can tell the script to break the loop. This saves electricity and prevents the model from diverging.

"""
