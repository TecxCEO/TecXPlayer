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
    self.filepath="../data/cube3x3/solution"
    self.max_steps = max_steps 
  def solve(self,given_state):
    self.current_state=given_state.copy()
    puzzle_data={
      "puzzle": {
        "puzzle_given": self.current_state,
        "puzzle_status":False,
        "moves_to_solve_puzzle":"",
        "moves_history": []
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
        self.update_nested_key(my_data["solution"],my_data["puzzle"]["puzzle_status"],my_data["puzzle"]["moves_to_solve_puzzle"],my_data["puzzle"]["moves_history"], data_batch, pk)
        print(f" pk in while loop = {pk} ")
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
  def update_nested_key(self,data,status,mtsp,p_moves_history=None,data_batch=None, pk = None):
    """
    Searches recursively for 'target_key' and updates its value.
    Works for both nested dictionaries and lists of dictionaries.
    """
    print(f"p_moves_history at start of loop = {p_moves_history} ")
    
    moves_history = []
    if pk is not None:
      if p_moves_history and len(p_moves_history)>pk and len(p_moves_history[pk])>1:
        moves_history = p_moves_history[pk]
      elif p_moves_history and len(p_moves_history) <= pk and (pk >0 and p_moves_history[pk - 1] and len(p_moves_history[pk - 1])>1):
        p_moves_history[pk] = []
    else:
      print(f" moves_history in else before  = {moves_history} ")
      moves_history = p_moves_history
      print(f" moves_history in else after = {moves_history} ")
    if moves_history is None:
      #moves_history = []
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
            moves_history = mh if len(mh) == 18 else None
            if moves_history and moves_history[-1] == self.max_steps:  # 16: ####
              moves_history[-1]= mh
              if isinstance(data_batch, str):
                data_batch = {}
              data_batch.update(data.copy())
            print(f"moves_history in if 20 = {moves_history} ")
          return data, p_moves_history, status, data_batch
      """
      if len(moves_history) == self.max_steps  and isinstance(moves_history[-1], list):
        if len(list(moves_history[self.max_steps-1])) in [18, 15]:
          self.delete_and_clean(data, moves_history)
      """
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
          print(f" current key is {key}, ")
          if len(data) ==19 and key!="state" and moves_history: #( moves_history and key == moves_history[0]):
            print(f" moves_history length = {len(moves_history)} ")
            print(f" moves_history[-1] in loop = {moves_history[-1]} ")
            if moves_history[-1] != self.max_steps:
              print(f" self.max_steps in loop = {self.max_steps} ")
            if key == moves_history[0] and len(moves_history) == self.max_steps and moves_history[-1] != self.max_steps:
              print(f" pk in loop after if = {pk} ")
              print(f" pk in loop before if 17 = {pk} ") if pk is not None else None
              if pk is not None and pk < 17 :
                print(f" pk in loop if 17 = {pk} ")
                pk +=1 
                print(f" pk in loop = {pk} ")
                if p_moves_history[pk] and len(p_moves_history[pk])>1 : # or ( len(p_moves_history[pk]) == 1 and p_moves_history[pk+1] is not exist )):
                  print(f" p_moves_history in loop if 17 = {p_moves_history} ")
                  print(f" p_moves_history[pk] in loop if 17 = {p_moves_history[pk]} ")
                  moves_history = p_moves_history[pk] 
                elif len(p_moves_history[pk]) == 1 and p_moves_history[pk+1] is not exist :
                  moves_history = [p_moves_history[pk]]
                elif not p_moves_history[pk] :
                  p_moves_history[pk] = []
                  moves_history = []
                print(f" The Key is being change from {key} to ")
                continue
          ######
          ######
          print(f" moves_history at if start.  = {moves_history} ")
          if key!="state" and (len(value) <=20 or len(data[key]) <= 20) and (len(value) >0 or len(data[key]) >0):
            data_batch.update({key:{}})
            if not moves_history or( isinstance(value, dict) and len(value) == 20):
              print("In the if for add key")
              if moves_history and moves_history[-1] == self.max_steps:
                moves_history[-2] = key
              elif not moves_history or (moves_history and moves_history[-1] != self.max_steps):
                moves_history += [key]
            if moves_history and ( len(moves_history) >1 and key != moves_history[0]) :
              removed_key = moves_history.pop(0)
            if moves_history and key == moves_history[0]:
              self.update_nested_key(value,status,mtsp,moves_history, data_batch[key])
            if locals().get("removed_key") :
              moves_history.insert(0, removed_key)
            print(f"I am here.")
            #if pk==0 and p_moves_history ==[] and len(moves_history)==18 :
            if pk is not None and pk >=0:
                p_moves_history[pk] = moves_history
            elif pk == None:
                p_moves_history= moves_history
            """
            if moves_history and len(moves_history)==self.max_steps and moves_history[-1] !=self.max_steps:
              if pk and pk >=0:
                p_moves_history[pk] = moves_history
              elif pk == Nome:
                p_moves_history[0] = moves_history
            elif moves_history and len(moves_history)<self.max_steps:
              if pk and pk >=0:
                p_moves_history[pk] = moves_history
                """
            print(f" moves_history at end before return = {moves_history} ")
            return data, p_moves_history, status, data_batch,pk
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
    time.sleep(11)
    #time.sleep(0.01) 
print(f" full_response = {full_response}")
