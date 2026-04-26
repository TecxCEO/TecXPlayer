import string
class EncodeDecode:
  def __init__(self, full_text):
    # here are all the unique characters that occur in this text
    # Define the components
    lowercase = string.ascii_lowercase          # a-z (26)
    uppercase = string.ascii_uppercase          # A-Z (26)
    digits = string.digits                      # 0-9 (10)
    special = """ !.,{"'}()[]:;?-\n"""
    # Combine them into one string
    chars = lowercase + uppercase + digits + special
    ##chars = lowercase + uppercase + digits + special + ''.join(chars)
    chars = sorted(list(set(chars)))
    # Create new stoi with special tokens
    chars = sorted(list(set(full_text))) + chars
    #chars = sorted(list(set(full_text)))
    # stoi = {}
    stoi['<PAD>'] = 0
    stoi['<SOS>'] = 1
    stoi['<EOS>'] = 2
    t= len(stoi)-1
    #stoi = {ch: i + 3 for i, ch in enumerate(chars)} # Shift everything by 3
    stoi = {ch: i+t for i, ch in enumerate(chars)} # Shift everything by 3
    itos = {i: ch for ch, i in stoi.items()} # Reverse map
    # encode=
    # decode=
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
    # Train and test splits
    ##data = torch.tensor(encode(text), dtype=torch.long)
  def encoder(self,str_in):
    return encode(str_in)
  def decoder(self, int_in):
    return decode(int_in)
    
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
def createTokens(self, full_text=full_text):
  t=len(stoi)-1
  if isinstance(full_text, dict):
    for key, value in (full_text):
      #if not self.stoi.get(key) and isinstance(value, dict):
      if not self.stoi.get(key) and key != "state" and isinstance(value, dict):
        str="'<"+key+">'"
        stoi[str] = (t+=1)
        self.createTokens(value)
        #if isinstance(value, dict):
          #self.createTokens(value)
      elif not isinstance(value, (dict, list)):
        if not self.stoi.get(key) and key != "state":
          stoi[key] = t+=1
        if  not self.stoi.get(value):
          stoi[value] = t+=1
    itos = {i: ch for ch, i in stoi.items()} # Reverse map
    # return stoi
    return stoi, itos
def createCharactarizedTokens(self, full_text=full_text):
  # In this method each and every word will be a list of character, like rgw = [21, 11, 27].
  #chars = sorted(list(set(full_text)))
  #stoi = {ch: i + 3 for i, ch in enumerate(chars)} # Shift everything by 3
  #stoi['<PAD>'] = 0
  #stoi['<SOS>'] = 1
  #stoi['<EOS>'] = 2
  #t=3+ len(chars)
  stoi = 
  stoi = 
  itos = {i: ch for ch, i in stoi.items()} # Reverse map
  return stoi, itos
