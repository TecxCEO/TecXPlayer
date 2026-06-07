# Cube3x3 traning dataset by Serial Sequential Move
import json
from cube3x3 import Cube3x3 as c3x3
import os
import sys
import time

class Solver(c3x3):
  def __init__(self):
    super().__init__()
    self.filename = "cube3x3trainingdataset.json"
    self.filepath="../data/cube3x3/solution"
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
    while True:
      print(f"While loop no = {(while_loop := while_loop + 1)}  are started.") ##########
      #print(f"\n\nwhile loop no = {while_loop }  are started.")
      # 1. Load your file
      with open(self.filename, "r") as rf:
        my_data = json.load(rf)
      #print(f"moves_history before start = {my_data["puzzle"]["moves_history"]}")
      if len(my_data["puzzle"]["moves_history"]) == 16:
        #print(f"data_batch before start = {data_batch}")
        #print(f"moves_history before start = {my_data["puzzle"]["moves_history"]}")
        data_batch = {}
      # 2. Update a key (no matter how deep it is)      
      if my_data["puzzle"]["puzzle_status"]==False:
        self.update_nested_key(my_data["solution"],my_data["puzzle"]["puzzle_status"],my_data["puzzle"]["moves_to_solve_puzzle"],my_data["puzzle"]["moves_history"], data_batch)
        with open(self.filename, "w") as wf:
          #json.dump(my_data, wf)
          json.dump(my_data, wf, indent=4)
      elif my_data["puzzle"]["puzzle_status"]==True:
        break
      #print(f"Move no = {(while_loop := while_loop + 1)}  are done.") ##########
      print(f"While loop no = {while_loop}  are done.") ##########
      ##############while_loop += 1
      if len(my_data["puzzle"]["moves_history"]) == 16:
        #print(f"data_batch after function call = {data_batch}")
        #print(f"moves_history after function call  = {my_data["puzzle"]["moves_history"]}")
        yield data_batch
  def delete_and_clean(self, data_to_process, moves_history, index=0):
    print(f" In The delete_and_clean function ")
    print(f" data_to_process at length = {len(data_to_process)}")
    print(f" data_to_process = {data_to_process}")
    if len(data_to_process) <=2 and (data_to_process.keys() in [[moves_history[index]], "state" ]):
      print(f"In The delete_and_clean function's if statement")
      print(f" data_to_process at length = {len(data_to_process)}")
      print(f" data_to_process = {data_to_process}")
      del data_to_process[moves_history[index]]
      return 
    elif len(data_to_process) > 2 and  index < len(moves_history)-1:
      print(f" data_to_process at length = {len(data_to_process)}")
      print(f" data_to_process = {data_to_process}")
      print(f" In The delete_and_clean function's elif statement")
      if index <  len(moves_history)-2:
        print(f" In The delete_and_clean function's elif's if ")
        print(f" data_to_process at length = {len(data_to_process)}")
        print(f" data_to_process = {data_to_process}")
        self.delete_and_clean(data_to_process[moves_history[index]], moves_history, index+1)
      elif index == len(moves_history)-2 and len(data_to_process[moves_history[index]]) in [16, 15] :
        print(f"  In The delete_and_clean function's elif's elif. ")
        print(f" data_to_process at length = {len(data_to_process)}")
        print(f" data_to_process = {data_to_process}")
        del data_to_process[moves_history[index]]
      return
  def update_nested_key(self,data,status,mtsp,moves_history=None,data_batch=None):
    """
    Searches recursively for 'target_key' and updates its value.
    Works for both nested dictionaries and lists of dictionaries.
    """
    ###
    if moves_history is None:
      moves_history = []
      status=False
    # If it's a dictionary, check keys or go deeper
    if isinstance(data, dict):
      if len(data)==20:
        print(f" In the if condition 20.")
        if all(key and len(value) not in [15,18,20] for key, value in data.items()):
          if moves_history and moves_history[-1] ==16:
            #print(f" moves_history ={moves_history}")
            #print(f" moves_history at length = {len(moves_history)}")
            #states,move_list,status=super().moves(data,mtsp,moves_history[0])
            states,move_list,status=super().moves(data,mtsp,[moves_history[-2]])
            #print(f" moves_history -2 ={moves_history[-2]}")
          else:
            #print(f" moves from else")
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
            #print(f" mh = {mh}")
            if moves_history and moves_history[-1] ==16:
              #moves_history += [mh]
              #moves_history[-1]= [mh]
              moves_history[-1]= mh
              #print(f" moves_history = {moves_history}")
              #print(f" moves_history at length = {len(moves_history)}")
              #print(f" moves_history[15] at length = {len(moves_history[-1])}")
              #print(f" moves_history[15] at length = {moves_history[-1]}")
              if isinstance(data_batch, str):
                data_batch = {}
              data_batch.update(data.copy())
          #print(f" moves_history ={moves_history}")
          return data, moves_history, status, data_batch
      if len(moves_history) == 16 and isinstance(moves_history[-1], list):
        print(f" moves_history at length = {len(moves_history)}")
        print(f" moves_history = {moves_history}")
        print(f" moves_history[15] at length = {len(moves_history[15])}")
        print(f" moves_history[15] at length = {moves_history[15]}")
        if len(list(moves_history[15])) in [18, 15]:
          self.delete_and_clean(data, moves_history)
      if len(data) < 20 and len(moves_history) <= 16: # and ( len(list(moves_history[15])) !> 1: and not isinstance(moves_history[-1], list)
        print(f" In the nested calling if condition.")
        if len(moves_history) ==14 and moves_history[-1] != 16 :
          moves_history += [15]
          moves_history += [16]
        if isinstance(data_batch, str):
                data_batch = {}
        data_batch.update({"state": data["state"]})
        for key, value in data.items():
          if key!="state" and (len(value) in [16,19,20] or len(data[key]) in [15,18,20]):
              print(f" key = {key}")
              data_batch.update({key:""})
              print(f" data_batch = {data_batch} ")
              if not moves_history or isinstance(value, dict) and len(value) == 20:
                print("In the if for add key")
                if moves_history and moves_history[-1] == 16:
                  print("In the if for add key by if.")
                  moves_history[-2] = key
                else:
                  print("In the if for add key by else")
                  moves_history += [key] 
              print(f"moves history for remove 0 key = {moves_history}")
              if moves_history and ( len(moves_history) >1 and key != moves_history[0]) :
                removed_key = moves_history.pop(0)
                print(f" Remove key from 0 = { removed_key }")
              print(f"moves_history before calling in the nested function = {moves_history}")
              print(f"moves_history length = {len(moves_history)}")
              if key == moves_history[0]:
                self.update_nested_key(value,status,mtsp,moves_history, data_batch[key])
              print(f" After function return, key = { key }")
              if locals().get("removed_key") :
                moves_history.insert(0, removed_key)
                print(f"moves_history after calling the nested function = {moves_history}")
                print(f"moves_history length = {len(moves_history)}")
              elif locals().get("removed_key"):
                print(f" Not added last time removed key, removed_key = { removed_key }")
              return
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
    time.sleep(0.01) 
print(f" full_response = {full_response}")
