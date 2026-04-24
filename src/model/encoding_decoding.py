# Create new stoi with special tokens
chars = sorted(list(set(full_text)))
stoi['<PAD>'] = 0
stoi['<SOS>'] = 1
stoi['<EOS>'] = 2
t= len(stoi)-1
#stoi = {ch: i + 3 for i, ch in enumerate(chars)} # Shift everything by 3
stoi = {ch: (t+=1) for i, ch in enumerate(chars)} # Shift everything by 3
itos = {i: ch for ch, i in stoi.items()} # Reverse map

# t=3+ len(chars)
# t= len(stoi)-1

"""
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
"""

# itos = {i: ch for ch, i in stoi.items()} # Reverse map
def get_nested_value(self, full_text):
  t=len(stoi)-1
    if isinstance(full_text, dict):
      for key, value in (full_text):
        str="'<"+key+">'"
        stoi[str] = t+=1
        if isinstance(value, dict):
          self.get_nested_value(value)
        elif isinstance(value, (dict, list)):
          stoi[value] = t+=1
      return stoi
def createTokens(self,full_text):
  #chars = sorted(list(set(full_text)))
  #stoi = {ch: i + 3 for i, ch in enumerate(chars)} # Shift everything by 3
  #stoi['<PAD>'] = 0
  #stoi['<SOS>'] = 1
  #stoi['<EOS>'] = 2
  #t=3+ len(chars)
  get_nested_value(self, full_text)
      
