# Cube3x3 traning dataset by Serial Sequential Move
import json
from cube3x3 import Cube3x3 as c3x3
import os
import sys

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
        json.dump(puzzle_data, f)
        #json.dump(puzzle_data, f, indent=4)
    while_loop=0
    data_batch = {}
    while True:
      # 1. Load your file
      with open(self.filename, "r") as rf:
        my_data = json.load(rf)
      if len(my_data["puzzle"]["moves_history"]) == 16:
        #### del data_batch
        data_batch = {}
        print(f"moves_history in loop = {my_data["puzzle"]["moves_to_solve_puzzle"]}")
      # 2. Update a key (no matter how deep it is)      
      if my_data["puzzle"]["puzzle_status"]==False:
        self.update_nested_key(my_data["solution"],my_data["puzzle"]["puzzle_status"],my_data["puzzle"]["moves_to_solve_puzzle"],my_data["puzzle"]["moves_history"], data_batch)
        with open(self.filename, "w") as wf:
          json.dump(my_data, wf)
          #json.dump(my_data, wf, indent=4)
      elif my_data["puzzle"]["puzzle_status"]==True:
        print( f"This Puzzle has been solved and The moves which were used to solve it, as followings")
        print(f"The moves for given puzzles solution ={my_data["puzzle"]["moves_to_solve_puzzle"]}")
        break
      print(f"Move no = {(while_loop := while_loop + 1)}  are done.")
      if len(my_data["puzzle"]["moves_history"]) == 16:
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
    print(f"moves_history = {moves_history}")
    ###
    if moves_history is None:
      moves_history = 3 #[]
      print(f"moves_history in if statement = {moves_history}")
      status=False
    # If it's a dictionary, check keys or go deeper
    if isinstance(data, dict):
      #print(f"data length={len(data)}")
      if len(data)==20:
        #print(f"so i am in if =20 condition")
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
              mh += move_list[i]
              #####data_batch.update({move_list[i]:states[i]}) ######
            if len(moves_history) == 15:
              moves_history += [mh]
              data_batch.update({data.copy()}) #####
          # return data, moves_history, status, moved_history
          return data, moves_history, status, data_batch
      if len(moves_history) == 16 and len(moves_history[15]) in [18, 15]:
        self.delete_and_clean(data, moves_history)
  
      # if (len(data)==16 or len(data)==19 or len(data)==20 )and len(moves_history) <16:
      if len(data) < 20 and len(moves_history) < 16:
        data_batch.update({"state": data["state"]})
        for key, value in data.items():
          if key!="state" and (len(value) in [16,19,20] or len(data[key]) in [15,18,20]):
            if (moves_history and moves_history[-1]!=key) or not moves_history:
              # if moved_history[key] and moved_history[key] is not in [None, ""]:
              # el
              ##########if moved_history is None:
                ###########moved_history.update({key:""})
              data_batch.update({key:""})
              # self.update_nested_key(value,status,mtsp,moves_history+[key], moved_history[key])
              print(f"data = {data}")
              print(f"moves_history = {moves_history}")
              self.update_nested_key(value,status,mtsp,moves_history+[key], data_batch[key])
              print(f"data = {data}")
              print(f"moves_history = {moves_history}")
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
  for char in s.solve(state_given_to_solve):
            sys.stdout.write(char+" ")
            sys.stdout.flush()
            full_response += (" " + char) # Collect for logging
            time.sleep(0.01) 
