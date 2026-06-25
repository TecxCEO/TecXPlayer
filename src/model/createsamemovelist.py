import json
move_paths=["<rgy>","<rgw>","<rgo>","<rby>","<rbw>","<rbo>","<grw>","<gry>","<grb>","<gow>","<goy>","<gob>","<yrg>","<yrb>","<yrw>","<yog>","<yob>","<yow>"]
mosf={'b':'g','g':'b','o':'r','r':'o','w':'y','y':'w'}
# 1. Three-Character Family Tokens (8 Families * 6 Flips = 48 Tokens)
same_move_list = {}
for moving_step in move_paths:
  same_move = None #[]
  f=moving_step.strip()[1]
  s=moving_step.strip()[2]
  t=moving_step.strip()[3]
  #if get(same_move) is not exist or None:
  same_move = [f"<{f}{s}{t}>"]
  #if locals.get(same_move):
  #same_move += f"<{f}{s}{t}>"
  same_move += [f"<{t}{s}{mosf[f]}>"]
  same_move += [f"<{mosf[f]}{s}{mosf[t]}>"]
  same_move += [f"<{mosf[t]}{s}{f}>"]
  same_move_list.update({same_move[0]: same_move})
  print(f" same move = {same_move}")
print(f" same move list = {same_move_list}")
print(f" same move list length = {len(same_move_list)}")
file= "list_of_same_move.json"
if not os.path.isfile(file):
  with open(file, "w") as f:
    json.dump(same_move_list, f, indent=4)
