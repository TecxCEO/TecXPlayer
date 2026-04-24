# Create new stoi with special tokens
chars = sorted(list(set(full_text)))
stoi = {ch: i + 3 for i, ch in enumerate(chars)} # Shift everything by 3

stoi['<PAD>'] = 0
stoi['<SOS>'] = 1
stoi['<EOS>'] = 2
t=3+ len(chars)
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi['<>'] = t+=1
stoi[''] = t+=1
stoi[''] = t+=1
stoi[''] = t+=1
stoi[''] = t+=1
stoi[''] = t+=1
stoi[''] = t+=1
stoi[''] = t+=1
stoi[''] = t+=1
stoi[''] = t+=1
stoi[''] = t+=1
stoi[''] = t+=1
stoi[''] = t+=1
stoi[''] = t+=1
stoi[''] = t+=1
stoi[''] = t+=1
stoi[''] = t+=1
stoi[''] = t+=1
stoi[''] = t+=1
stoi[''] = t+=1
stoi[''] = t+=1
stoi[''] = t+=1
stoi[''] = t+=1
stoi[''] = t+=1
stoi[''] = t+=1
stoi[''] = t+=1
stoi[''] = t+=1
stoi[''] = t+=1
stoi[''] = t+=1
stoi[''] = t+=1
stoi[''] = t+=1
stoi[''] = t+=1
stoi[''] = t+=1
stoi[''] = t+=1
stoi[''] = t+=1
stoi[''] = t+=1
stoi[''] = t+=1
stoi[''] = 
stoi[''] = 
stoi[''] = 
stoi[''] = 
stoi[''] = 
stoi[''] = 
stoi[''] = 
stoi[''] = 
stoi[''] = 
stoi[''] = 
stoi[''] = 
stoi[''] = 
stoi[''] = 
stoi[''] = 
stoi[''] = 
stoi[''] = 
stoi[''] = 
stoi[''] = 
stoi[''] = 
stoi[''] = 
stoi[''] = 
stoi[''] = 
stoi[''] = 
stoi[''] = 
stoi[''] = 
stoi[''] = 
stoi[''] = 
stoi[''] = 
stoi[''] = 
stoi[''] = 
stoi[''] = 
stoi[''] = 

itos = {i: ch for ch, i in stoi.items()} # Reverse map

