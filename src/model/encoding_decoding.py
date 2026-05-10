import string
class EncodeDecode:
  def __init__(self, full_text = None):
    self.full_text = full_text
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
    #print(f"full_text={full_text}")
    ##if full_text:
      ##chars = sorted(list(set(full_text))) + chars
    #chars = sorted(list(set(full_text)))
    #self.stoi = {}
    
    #####t= len(self.stoi)-1
    #stoi = {ch: i + 3 for i, ch in enumerate(chars)} # Shift everything by 3
    self.stoi = {ch: i for i, ch in enumerate(chars, start=3)} # Shift everything by 3
    self.stoi['<PAD>'] = 0
    self.stoi['<SOS>'] = 1
    self.stoi['<EOS>'] = 2
    self.itos = {i: ch for ch, i in self.stoi.items()} # Reverse map
    print(f" stoi = {self.stoi}")
    print(f" stoi = {len(self.stoi)}")
    print(f" itos = {self.itos}")
    print(f" itos = {len(self.itos)}")
    # encode=
    # decode=
    self.encode = lambda s: [self.stoi[c] for c in s] # encoder: take a string, output a list of integers
    
    ##########self.decode = lambda l: ([self.itos[i] for i in l]) # decoder: take a list of integers, output a string
    self.decode = lambda l: ''.join([self.itos[i] for i in l]) # decoder: take a list of integers, output a string
    ####self.stoi = stoi
    # Train and test splits
    ##data = torch.tensor(encode(text), dtype=torch.long)
  
  def encoder(self, str_in):
    ##s= str_in ###
    ##if self.stoi[str_in]:
    if self.encode(str_in):
      # Returns the value if found, otherwise returns None
      ##encode = lambda str_in: self.stoi.get(str_in)
      #return (
      #encode = lambda str_in: [self.stoi[str_in] if self.stoi[str_in] else continue]
      return self.encode(str_in)
    elif not self.encode(str_in) and not isinstance(str_in, (dict, list)):
      self.createTokens(str_in)
      return self.encode(str_in)
    elif isinstance(str_in, (dict, list)):
      for strin in str_in:
        if isinstance(strin, (dict, list)):
          if self.encode(strin):
            return self.encode(strin)
          else:
            self.encoder(strin) 
      #encoder(self, strin) for strin in str_in
      # createTokens(str_in)
      # return self.encode(str_in)
    # elif 
  # return encode = lambda s: [self.stoi[c] if self.stoi[c] else createTokens(c) for c in str_in]
  
    # return encode
  #######
  #def decoder(self, int_in):
    #if self.itos[int_in]:
      ##return self.decode(int_in)
    #else:
      # createTokens(str_in)
      # return self.decode(int_in)
  def return_stoi_size(self):
    return len(self.stoi)
  # def createTokens(self, full_text = self.full_text):
  #def createTokens(self, full_text, stoi, itos):
  def createTokens(self, full_text):
    self.stoi = stoi
    self.itos = itos
    print(f"stoi len before token cretion = {len(self.stoi)}")
    if isinstance(full_text, dict):
      print(f" full_text = {len(full_text)}")
      for key, value in full_text.items():
        #if not self.stoi.get(key) and isinstance(value, dict):
        # if not self.stoi.get(key) and key != "state" and isinstance(value, dict):  
        str="<"+key+">"
        #if not self.stoi.get(key) and isinstance(value, dict):
        if not self.stoi.get(str) and isinstance(value, dict):
          t=len(self.stoi) ####
          self.stoi[str] = t
          #if len(value) != 20 and len(value) in (16, 19):
        if isinstance(value, dict) and len(value) in (16, 19, 20):
          # This works! It checks if len(value) is 16, 19, or 20.
          print(f"deep dive in dict of {key} of {len(value)} len value.\n")
          #self.stoi, self.itos = self.createTokens(value, self.stoi, self.itos)
          self.createTokens(value)
          #####self.createTokens(value, self.stoi, self.itos)
        elif not isinstance(value, (dict, list)):
          if not self.stoi.get(key) and key != "state":
            t=len(self.stoi) ######
            self.stoi[key] = t
          if  not self.stoi.get(value):
            t=len(self.stoi)
            self.stoi[value] = t
      ####print(f"stoi len = {len(self.stoi)}")
      ##print(f"stoi = {self.stoi}")
      ##print(f"itos = {self.itos}")
      print(f"itos len = {len(self.itos)}")
      self.itos = {i: ch for ch, i in self.stoi.items()}
      ##print(f"itos at the end of createTokens in encoding decoding = {self.itos}")
      print(f"itos len at the end of createTokens in encoding decoding = {len(self.itos)}")
      ##print(f"stoi at the end of createTokens in encoding decoding = {self.stoi}")
      print(f"stoi len at the end of createTokens in encoding decoding = {len(self.stoi)}")
      return self.stoi, self.itos


# t=3+ len(chars)
# t= len(stoi)-1

# itos = {i: ch for ch, i in stoi.items()} # Reverse map
###def createCharactarizedTokens(self, full_text=full_text):
  # In this method each and every word will be a list of character, like rgw = [21, 11, 27].
  #chars = sorted(list(set(full_text)))
  #stoi = {ch: i + 3 for i, ch in enumerate(chars)} # Shift everything by 3
  #stoi['<PAD>'] = 0
  #stoi['<SOS>'] = 1
  #stoi['<EOS>'] = 2
  #t=3+ len(chars)
  ###stoi = 
  ###stoi = 
  ###itos = {i: ch for ch, i in stoi.items()} # Reverse map
  ###return stoi, itos
if __name__ == "__main__":
  print(f"At start")
  import json
  file= "data/dataset/cube3x3solvingdataset.json"
  with open(file, 'r') as f:
            data = json.load(f)
  edc = EncodeDecode(data['solution'])
  result=edc.createTokens(data["solution"])
  print(f" Result= {result}\n")
  print(f"stoi = {edc.stoi}\n\n")
  print(f"itos = {edc.itos}")
