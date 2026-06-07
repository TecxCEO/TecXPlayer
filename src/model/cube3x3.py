import json

#class CubeSolver:
class Cube3x3:
  def __init__(self):
    self.output_file="./puzzle_move_and_states.json"
    faces={'b':'blue','g':'green','o':'orange','r':'red','y':'yellow','w':'white'}
    mutually_oppsite_side_faces={'blue':'green','orange':'red','yellow':'white'}
    self.mosf={'b':'g','g':'b','o':'r','r':'o','y':'w','w':'y'}
    colors={'blue','green','orange','red','yellow','white'}
    # states={}
    self.move_paths=["rgy","rgw","rgo","rby","rbw","rbo","grw","gry","grb","gow","goy","gob","yrg","yrb","yrw","yog","yob","yow"]
    vertex={
      "rgy":["red","green","yellow"],
      "rgw":["red","green","white"],
      "rby":["red","blue","yellow"],
      "rbw":["red","blue","white"],
      "ogy":["orange","green","yellow"],
      "ogw":["orange","green","white"],
      "oby":["orange","blue","yellow"],
      "obw":["orange","blue","white"]
    }
    edges={
      "rb":["red","blue"],
      "rg":["red","green"],
      "rw":["red","white"],
      "ry":["red","yellow"],
      "ob":["orange","blue"],
      "og":["orange","green"],
      "ow":["orange","white"],
      "oy":["orange","yellow"],
      "by":["blue","yellow"],
      "bw":["blue","white"],
      "gw":["green","white"],
      "gy":["green","yellow"]
    }
    self.solution={
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
    state_given={
      "rgy":"ogw",
      "rgw":"ybo",
      "rby":"ryg",
      "rbw":"bwr",
      "ogy":"yrb",
      "ogw":"oyg",
      "oby":"owb",
      "obw":"wrg",
      "rb":"gy",
      "rg":"rw",
      "rw":"yr",
      "ry":"by",
      "ob":"gw",
      "og":"bw",
      "ow":"oy",
      "oy":"ow",
      "by":"go",
      "bw":"rb",
      "gw":"ob",
      "gy":"gr"
    }
  def mover(self,moving_step,state_given):
    state=state_given.copy()
    f=moving_step.strip()[0]
    s=moving_step.strip()[1]
    cc=moving_step.strip()[2]
    if f!= self.mosf[cc]and f!=cc:
      c=moving_step.strip()[2]
    elif f == self.mosf[cc]and f!=cc:
      for mosf in self.mosf:
        if mosf != f and mosf != s and mosf != self.mosf[f] and mosf != self.mosf[s]:
          c=mosf
          break
    moving_block=[]
    mb={}
    moving_block.append(f"{f}{s}{c}")
    moving_block.append(f"{f}{s}{self.mosf[c]}")
    moving_block.append(f"{self.mosf[f]}{s}{c}")
    moving_block.append(f"{self.mosf[f]}{s}{self.mosf[c]}")
    moving_block.append(f"{f}{s}")
    moving_block.append(f"{s}{c}")
    moving_block.append(f"{self.mosf[f]}{s}")
    moving_block.append(f"{s}{self.mosf[c]}")
    for name in moving_block:
      st_e=""
      for state_element in state:
        if sorted(name) == sorted(state_element):
          for n in range(len(name)):
            for se in range(len(state_element)):
              if state_element.strip()[se]==name.strip()[n]:
                st_e+=state[state_element].strip()[se]
          if st_e!="":
            mb.update({name:st_e})
    mbc=mb.copy()
    if self.mosf[f]!=cc:
      mb[f"{f}{s}{c}"]=mbc[f"{f}{s}{self.mosf[c]}"]
      mb[f"{s}{c}"]=mbc[f"{f}{s}"]
      mb[f"{self.mosf[f]}{s}{c}"]=mbc[f"{f}{s}{c}"]
      mb[f"{self.mosf[f]}{s}"]=mbc[f"{s}{c}"]
      mb[f"{self.mosf[f]}{s}{self.mosf[c]}"]=mbc[f"{self.mosf[f]}{s}{c}"]
      mb[f"{s}{self.mosf[c]}"]=mbc[f"{self.mosf[f]}{s}"]
      mb[f"{f}{s}{self.mosf[c]}"]=mbc[f"{self.mosf[f]}{s}{self.mosf[c]}"]
      mb[f"{f}{s}"]=mbc[f"{s}{self.mosf[c]}"]
    elif self.mosf[f]==cc:
      mb[f"{self.mosf[f]}{s}{c}"]=mbc[f"{f}{s}{self.mosf[c]}"]
      mb[f"{self.mosf[f]}{s}"]=mbc[f"{f}{s}"]
      mb[f"{self.mosf[f]}{s}{self.mosf[c]}"]=mbc[f"{f}{s}{c}"]
      mb[f"{s}{self.mosf[c]}"]=mbc[f"{s}{c}"]
      mb[f"{f}{s}{self.mosf[c]}"]=mbc[f"{self.mosf[f]}{s}{c}"]
      mb[f"{f}{s}"]=mbc[f"{self.mosf[f]}{s}"]
      mb[f"{f}{s}{c}"]=mbc[f"{self.mosf[f]}{s}{self.mosf[c]}"]
      mb[f"{s}{c}"]=mbc[f"{s}{self.mosf[c]}"]
    for name in moving_block:
      mb_e=""
      for state_element in state:
        if sorted(name) == sorted(state_element):
          for se in range(len(state_element)):
            for n in range(len(name)):
              if state_element.strip()[se]==name.strip()[n]:
                mb_e+=mb[name].strip()[n]
          if mb_e!="":
            state.update({state_element:mb_e})
    return state
  def moves(self, state_given_to_solve,mtsp,move_history=""):
    moves_to=list(self.move_paths)
    cur_state=state_given_to_solve.copy()
    i=0
    states = []
    moved_options_list=[]
    puzzle_solve= False
    last_move=""
    move_path_history=list(move_history) if move_history else move_history
    while cur_state and i<len(moves_to):
      if move_path_history!="" and move_path_history:
        last_move=move_path_history[-1]
        print(f"last move from moves funcion of cube3x3 = {last_move}")
      if last_move.strip()[:2]!=moves_to[i].strip()[:2] or not last_move:
        states+= [self.mover(moves_to[i],cur_state)]
        moved_options_list+=[moves_to[i]]
      #if states and states[-1]==self.solution:
        #puzzle_solve=True
        #mtsp=move_path_history+moved_options_list[i]
        #return states[i], moved_options_list[i], puzzle_solve
      i=i+1
    return states, moved_options_list, puzzle_solve
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
  mtsp=[]
  c3x3=Cube3x3()
  result=c3x3.moves(state_given_to_solve,mtsp)
  print(result)
