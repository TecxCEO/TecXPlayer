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
      ####print(f"while loop no = {while_loop }  are started.")
      # 1. Load your file
      with open(self.filename, "r") as rf:
        my_data = json.load(rf)
      print(f"moves_history before start = {my_data["puzzle"]["moves_history"]}")
      if len(my_data["puzzle"]["moves_history"]) == 16:
        #### del data_batch
        print(f"data_batch before start = {data_batch}")
        print(f"moves_history before start = {my_data["puzzle"]["moves_history"]}")
        data_batch = {}
      # 2. Update a key (no matter how deep it is)      
      if my_data["puzzle"]["puzzle_status"]==False:
        self.update_nested_key(my_data["solution"],my_data["puzzle"]["puzzle_status"],my_data["puzzle"]["moves_to_solve_puzzle"],my_data["puzzle"]["moves_history"], data_batch)
        with open(self.filename, "w") as wf:
          #json.dump(my_data, wf)
          json.dump(my_data, wf, indent=4)
      elif my_data["puzzle"]["puzzle_status"]==True:
        break
      #print(f"Move no = {(while_loop := while_loop + 1)}  are done.")
      while_loop += 1
      #print(f"my_data[puzzle][moves_history] len = {len(my_data["puzzle"]["moves_history"])}")
      if len(my_data["puzzle"]["moves_history"]) == 16:
        print(f"data_batch after function call = {data_batch}")
        print(f"moves_history after function call  = {my_data["puzzle"]["moves_history"]}")
        yield data_batch
  def delete_and_clean(self, data_to_process, moves_history, index=0):
    if len(data_to_process) <=2 and (data_to_process.keys() in [[moves_history[index]], "state" ]):
      del data_to_process[moves_history[index]]
      return 
    elif len(data_to_process) > 2 and  index < len(moves_history)-1:
      if index <  len(moves_history)-2:
        delete_and_clean(data_to_process[moves_history[index]], moves_history, index+1)
      elif index == len(moves_history)-2 and len(data_to_process[moves_history[index]]) in [16, 15] :
        del data_to_process[moves_history[index]]
      return
      

  #####def update_nested_key(self,data,status,mtsp,moves_history=None,moved_history=None):
  def update_nested_key(self,data,status,mtsp,moves_history=None,data_batch=None):
    """
    Searches recursively for 'target_key' and updates its value.
    Works for both nested dictionaries and lists of dictionaries.
    """
    ####print(f"moves_history at starting = {moves_history}")
    ###
    if moves_history is None:
      moves_history = []
      ####print(f"moves_history was None in if statement now = {moves_history}")
      status=False
    # If it's a dictionary, check keys or go deeper
    if isinstance(data, dict):
      #print(f"data length={len(data)}")
      if len(data)==20:
        print(f" In the if condition 20.")
        ####print(f"moves_history in isinstance statement = {moves_history}")
        if all(key and len(value) not in [15,18,20] for key, value in data.items()):
          states,move_list,status=super().moves(data,mtsp,moves_history)
          #print(f"moves_history={moves_history}")
          data.update({"state":data.copy()})
          for dic_key in list(data.keys()):
            dic_value=data[dic_key]
            if not isinstance (dic_value,(dict,list)) and dic_key != "state":
              del data[dic_key]
          #if len(states) in [15,18] and len(move_list) in [15,18] and status is False:
          if len(states) in [1,15,18] and len(move_list) in [1,15,18]:
            #print(f"data length= {len(data)}")
            mh = []
            for i in range(len(states)):
              data.update({move_list[i]:states[i]})
              mh += [move_list[i]]
            print(f" mh = {mh}")
              #####data_batch.update({move_list[i]:states[i]}) ######
            if len(moves_history) == 15:
              moves_history += [mh]
              print(f" moves_history = {moves_history}")
              print(f" moves_history at length = {len(moves_history)}")
              print(f" moves_history = {moves_history}")
              print(f" moves_history[15] at length = {len(moves_history[15])}")
              print(f" moves_history[15] at length = {moves_history[15]}")
              if isinstance(data_batch, str):
                data_batch = {}
              data_batch.update(data.copy())
              # data_batch.update(data.copy()) #####
          # return data, moves_history, status, moved_history
          return data, moves_history, status, data_batch
      if len(moves_history) == 16 and len(moves_history[15]) in [18, 15]:
        print(f" moves_history at length = {len(moves_history)}")
        print(f" moves_history = {moves_history}")
        print(f" moves_history[15] at length = {len(moves_history[15])}")
        print(f" moves_history[15] at length = {moves_history[15]}")
        self.delete_and_clean(data, moves_history)
  
      # if (len(data)==16 or len(data)==19 or len(data)==20 )and len(moves_history) <16:
      if len(data) < 20 and len(moves_history) < 16:
        print(f" In the nested calling if condition.")
        #### print(f" data_batch = { data_batch}")
        if isinstance(data_batch, str):
                data_batch = {}
        data_batch.update({"state": data["state"]})
        #items_list = list(data.items())
        items_list = list((data.items()).copy)
        #for key, value in data.items():
        for key, value in items_list:
          if key!="state" and (len(value) in [16,19,20] or len(data[key]) in [15,18,20]):
            if (moves_history and moves_history[-1]!=key) or not moves_history:
              print(f" key = {key}")
              # if moved_history[key] and moved_history[key] is not in [None, ""]:
              # el
              ##########if moved_history is None:
                ###########moved_history.update({key:""})
              data_batch.update({key:""})
              # self.update_nested_key(value,status,mtsp,moves_history+[key], moved_history[key])
              print(f"data_batch = {data_batch}")
              moves_history += [key]
              ######print(f"moves_history before calling in the nested function = {moves_history}")
              
              # self.update_nested_key(value,status,mtsp,moves_history+[key], data_batch[key])
              self.update_nested_key(value,status,mtsp,moves_history, data_batch[key])
              #### print(f"data after nested calling = {data}")
              ####print(f"moves_history after nested calling= {moves_history}")
              return ####
            #if status == True and mtsp:
              #print(f"mtsp={mtsp}")
              #return
        #print(f"rec loop end.")
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
  # result=s.solve(state_given_to_solve)
  # print(result)
  full_response = []
  for char in s.solve(state_given_to_solve):
    # sys.stdout.write(char+" ")
    sys.stdout.write(str(char) + " ")
    sys.stdout.flush()
    full_response += (" " + str(char)) # Collect for logging
    time.sleep(0.01) 
print(f" full_response = {full_response}")
