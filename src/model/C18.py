# Cube3x3 traning dataset by Serial Sequential Move
import json
from cube3x3 import Cube3x3 as c3x3
import os
import sys
import time

class Solver(c3x3):
  def __init__(self, max_steps = 20, batch = 18):
    # def __init__(self):
    super().__init__()
    self.filename = "cube3x3dataset20steps18batch.json"
    # self.filename = "cube3x3trainingdataset.json"
    self.filepath="./data/cube3x3/solution"
    self.max_steps = max_steps 
  def solve(self,given_state):
    self.current_state=given_state.copy()
    puzzle_data={
      "puzzle": {
        "puzzle_given": self.current_state,
        "puzzle_status":False,
        "moves_to_solve_puzzle":"",
        "moves_history": []
        "p_moves_history": []
      },
      "solution":self.current_state
    }
    if not os.path.isfile(self.filename):
      with open(self.filename, "w") as f:
        #json.dump(puzzle_data, f)
        json.dump(puzzle_data, f, indent=4)
    while_loop=0
    data_batch = {}
    print(f"while loop is going to started.")
    pk = 0
    while True:
      print(f"While loop no = {(while_loop := while_loop + 1)}  are started.") ##########
      #print(f"\n\nwhile loop no = {while_loop }  are started.")
      # 1. Load your file
      with open(self.filename, "r") as rf:
        my_data = json.load(rf)
      if len(my_data["puzzle"]["moves_history"]) == self.max_steps:
        data_batch = {}
      # 2. Update a key (no matter how deep it is)      
      if my_data["puzzle"]["puzzle_status"]==False:
        #self.update_nested_key(my_data["solution"],my_data["puzzle"]["puzzle_status"],my_data["puzzle"]["moves_to_solve_puzzle"],my_data["puzzle"]["moves_history"], data_batch, pk)
        #self.update_nested_key(my_data["solution"],my_data["puzzle"]["puzzle_status"],my_data["puzzle"]["moves_to_solve_puzzle"],my_data["puzzle"]["moves_history"], data_batch, pk,my_data["puzzle"]["moves_history"])
        self.update_nested_key(my_data["solution"],my_data["puzzle"]["puzzle_status"],my_data["puzzle"]["moves_to_solve_puzzle"],my_data["puzzle"]["moves_history"], data_batch, pk,my_data["puzzle"]["p_moves_history"])
        print(f" pk in while loop = {pk} ")
        print(f" my_data[puzzle][moves_history] in while loop = {my_data["puzzle"]["moves_history"]} ")
        #print(f" my data = { my_data}")
        with open(self.filename, "w") as wf:
          #json.dump(my_data, wf)
          json.dump(my_data, wf, indent=4)
      elif my_data["puzzle"]["puzzle_status"]==True:
        break
      #print(f"Move no = {(while_loop := while_loop + 1)}  are done.") ##########
      print(f"While loop no = {while_loop}  are done.") ##########
      ##############while_loop += 1
      if len(my_data["puzzle"]["moves_history"]) == self.max_steps:
        yield data_batch
  def delete_and_clean(self, data_to_process, moves_history, index=0):
    if len(data_to_process) >= 2 and  index < len(moves_history)-1:
      if index <  len(moves_history)-2:
        self.delete_and_clean(data_to_process[moves_history[index]], moves_history, index+1)
        if data_to_process[moves_history[index]] and len(data_to_process[moves_history[index]]) == 1 and next(iter(data_to_process[moves_history[index]])) == "state":# data_to_process[moves_history[index]].keys == "state" 
          del data_to_process[moves_history[index]]
          del moves_history[index]
      elif index == len(moves_history)-2 and len(data_to_process[moves_history[index]]) in [16, 15] :
        del data_to_process[moves_history[index]]
        del moves_history[index+1]
        del moves_history[index]
      return
  #def update_nested_key(self,data,status,mtsp,p_moves_history=None,data_batch=None, pk = None):
  def update_nested_key(self,data,status,mtsp,moves_history=None,data_batch=None, pk = None,p_moves_history=None):
    """
    Searches recursively for 'target_key' and updates its value.
    Works for both nested dictionaries and lists of dictionaries.
    """
    print(f"p_moves_history at start of loop = {p_moves_history} ")
    
    moves_history = []
    print (f" pk = {pk}")
    if pk is not None:
      if moves_history:
        del moves_history
      if pk>=0:
        # if not locals().get(p_moves_history[pk]) :
        if pk >= len(p_moves_history): # or (pk < len(p_moves_history) and not p_moves_history[pk] ):
          p_moves_history += []
          #p_moves_history[pk] = []
          # print(f"p_moves_history[{pk}] = {p_moves_history[pk]} ")
        elif p_moves_history and pk < len(p_moves_history) and p_moves_history[pk]:
          # elif p_moves_history and pk < len(p_moves_history) and locals().get(p_moves_history[pk]):
          if isinstance(p_moves_history[pk], str) and len(p_moves_history[pk])==3:
            ####moves_history = [p_moves_history[pk]]
            print(f"moves_history from pk not none and str  = {moves_history} ")
          elif isinstance(p_moves_history[pk], list) and len(p_moves_history[pk])>1:
            moves_history.extend(p_moves_history[pk])
            print(f"moves_history from pk not none and list  = {moves_history} ")
          print(f"DEBUG: pk value is {pk}, list total length is {len(p_moves_history)}")
          print(f"p_moves_history[{pk}] = {p_moves_history[pk]}")
      else:
        #moves_history = p_moves_history
        moves_history.extend(p_moves_history)
        print(f"p_moves_history from else = {p_moves_history} ")
        ######moves_history = p_moves_history[pk] if p_moves_history and locals().get(p_moves_history[pk]) else p_moves_history
    if pk is None:
      #moves_history = p_moves_history
      moves_history.extend(p_moves_history)
      print(f"moves_history from if pk is None")
    print(f"moves_history after get value from p_m = {moves_history} ")
    print(f" moves_history in else after = {moves_history} ")
    if moves_history is None:
      moves_history = []
      status=False
    if isinstance(data, dict):
      if len(data)==20:
        if all(key and len(value) not in [15,18,20] for key, value in data.items()):
          if moves_history and moves_history[-1] == self.max_steps: # 16: ####
            states,move_list,status=super().moves(data,mtsp,[moves_history[-2]])
          else:
            states,move_list,status=super().moves(data,mtsp,moves_history)
          data.update({"state":data.copy()})
          for dic_key in list(data.keys()):
            dic_value=data[dic_key]
            if not isinstance (dic_value,(dict,list)) and dic_key != "state":
              del data[dic_key]
          if len(states) in [1,15,18] and len(move_list) in [1,15,18]:
            mh = []
            for i in range(len(states)):
              data.update({move_list[i]:states[i]})
              mh += [move_list[i]]
            
            #p_moves_history = mh if len(mh) == 18 else None
            if len(mh) == 18 :
              #p_moves_history = mh
              p_moves_history.extend(mh)
              print(f"p_moves_history  = {p_moves_history} at first move it's length = {len(p_moves_history)} ")
            if moves_history and moves_history[-1] == self.max_steps:  # 16: ####
              moves_history[-1]= mh
              if isinstance(data_batch, str):
                data_batch = {}
              data_batch.update(data.copy())
              #p_moves_history[-1]= mh
          print(f"p_moves_history in if 20 = {p_moves_history} ")
          #return data, p_moves_history, status, data_batch
          return data,status,mtsp,moves_history,data_batch,pk,moves_history
      if pk and pk == 17:
        while pk >=0:
          moves_history = p_moves_history[pk]
          if len(moves_history) == self.max_steps  and isinstance(moves_history[-1], list):
            if len(list(moves_history[self.max_steps-1])) in [18, 15]:
              self.delete_and_clean(data, moves_history)
              pk -= 1
            else:
              break
        
      if len(data) < 20 and len(moves_history) <= self.max_steps:
        if len(moves_history) == self.max_steps -2 and moves_history[-1] != self.max_steps :
          moves_history += [self.max_steps -1]
          moves_history += [self.max_steps]
        data_batch.update({"state": data["state"]})
        for key, value in data.items():
          ######
          ######
          #print(f" current key is {key}, ")
          if len(data) ==19 and key!="state" and moves_history: #( moves_history and key == moves_history[0]):
            #print(f" moves_history length = {len(moves_history)} ")
            #print(f" moves_history[-1] in loop = {moves_history[-1]} ")
            ##if moves_history[-1] != self.max_steps:
              ##print(f" self.max_steps in loop = {self.max_steps} ")
            ####if key == moves_history[0] and len(moves_history) == self.max_steps and moves_history[-1] != self.max_steps:
            if key == moves_history[0] and len(moves_history) == self.max_steps and moves_history[-1] != self.max_steps and (isinstance(moves_history[-1], list) and len(moves_history[-1])==15 ):
              #print(f
              #print(f" pk in loop after if = {pk} ")
              #print(f" pk in loop before if 17 = {pk} ") if pk is not None else None
              if pk is not None and pk < 17 :
                #print(f" pk in loop if 17 = {pk} ")
                p_moves_history[pk] = moves_history
                pk +=1 
                print(f" pk in loop = {pk} ")
                if moves_history:
                  del moves_history
                #if p_moves_history[pk]:
                  #moves_history.extend(p_moves_history[pk])
                  #moves_history = p_moves_history[pk]
                # el
                #if not locals().get(p_moves_history[pk]) :
                moves_history = []
                if pk >= len(p_moves_history) : #or (pk < len(p_moves_history) and not p_moves_history[pk]) :
                  p_moves_history += []
                  # p_moves_history[pk] = []
                moves_history = p_moves_history[pk]
                #print(f" The Key is being change from {key} to ")
                continue
          ######
          print(f" moves_history at if start for add key.  = {moves_history} ")
          if key!="state" and (len(value) <=20 or len(data[key]) <= 20) and (len(value) >0 or len(data[key]) >0):
            data_batch.update({key:{}})
            if not moves_history or( isinstance(value, dict) and len(value) == 20):
              print("In the if for add key")
              if moves_history and moves_history[-1] == self.max_steps:
                moves_history[-2] = key
              elif not moves_history or (moves_history and moves_history[-1] != self.max_steps):
                moves_history += [key]
              print(f" moves_history after added key.  = {moves_history} ")
              #
            if moves_history and ( len(moves_history) >1 and key != moves_history[0]) :
              removed_key = moves_history.pop(0)
            print(f" moves_history before nested calling  = {moves_history} ")
            print(f" p_moves_history before = {p_moves_history} ")
            print(f" data_batch[] before nested calling = {data_batch} ")
            if moves_history and key == moves_history[0]:
              self.update_nested_key(value,status,mtsp,moves_history, data_batch[key])
            print(f" data_batch[{key}] after nested calling = {data_batch[key]} ")
            #print(f" p_moves_history aft= {p_moves_history} ")
            print(f" moves_history after nested calling  = {moves_history} ")
            print(f" p_moves_history after nested calling  = {p_moves_history} ")
            
            if locals().get("removed_key") :
              moves_history.insert(0, removed_key)
            print(f" p_moves_history aft= {p_moves_history} ")
            print(f" moves_history after nested calling  = {moves_history} ")
            print(f"I am here.")
            if pk is not None:
              if pk>=0:
                #if not locals().get(p_moves_history[pk]) :
                if pk >= len(p_moves_history) : #or (pk < len(p_moves_history) and not p_moves_history[pk] ):
                  p_moves_history += []
                  # p_moves_history[pk] = []
                  #p_moves_history[pk] = moves_history
              p_moves_history[pk] = moves_history
            elif pk is None:
              ####p_moves_history = moves_history
              del p_moves_history
              p_moves_history = []
              p_moves_history.extend(moves_history)
              print(f" p_moves_history in if none by extend= {p_moves_history} ")
            print(f" moves_history at end before return = {moves_history} ")
            print(f" p_moves_history at end before return = {p_moves_history} ")
            #return data, p_moves_history, status, data_batch,pk
            time.sleep(1)
            #return
            if pk is not None :
              return data,status,mtsp,p_moves_history,data_batch,pk
            elif pk is None :
              return data,status,mtsp,p_moves_history,data_batch
            
        return
if __name__=="__main__":
  state_given_to_solve={
      "rgy":"rgy",
      "rgw":"rgw",
      "rby":"rby",
      "rbw":"rbw",
      "ogy":"ogy",
      "ogw":"ogw",
      "oby":"oby",
      "obw":"obw",
      "rb":"rb",
      "rg":"rg",
      "rw":"rw",
      "ry":"ry",
      "ob":"ob",
      "og":"og",
      "ow":"ow",
      "oy":"oy",
      "by":"by",
      "bw":"bw",
      "gw":"gw",
      "gy":"gy"
    }
  s=Solver()
  full_response = []
  for char in s.solve(state_given_to_solve):
    sys.stdout.write(str(char) + " ")
    sys.stdout.flush()
    full_response += (" " + str(char)) # Collect for logging
    time.sleep(4)
    #time.sleep(0.01) 
print(f" full_response = {full_response}")
