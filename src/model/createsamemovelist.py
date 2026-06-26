import json
move_paths=["<rgy>","<rgw>","<rgo>","<rby>","<rbw>","<rbo>","<grw>","<gry>","<grb>","<gow>","<goy>","<gob>","<yrg>","<yrb>","<yrw>","<yog>","<yob>","<yow>"]
mosf={'b':'g','g':'b','o':'r','r':'o','w':'y','y':'w'}
# 1. Three-Character Family Tokens (8 Families * 6 Flips = 48 Tokens)
same_move_list = {}
a = None
b = None
c = None
for moving_step in move_paths:
  same_move = None #[]
  f=moving_step.strip()[1]
  s=moving_step.strip()[2]
  t=moving_step.strip()[3]
  if f != mosf[t]:
    if b is None or b not in [t, mosf[t]]:
      a = f
      b = t
    if b == mosf[t] and a == f:
      c = t
    same_move = [f"<{f}{s}{t}>"]
    same_move += [f"<{t}{s}{mosf[f]}>"]
    same_move += [f"<{mosf[f]}{s}{mosf[t]}>"]
    same_move += [f"<{mosf[t]}{s}{f}>"]
  elif f == mosf[t]:
    if a == f and b == mosf[c] and a not in [b, c] and s not in [b, c]:
      same_move = [f"<{f}{s}{t}>"]
      same_move += [f"<{b}{s}{c}>"]
      same_move += [f"<{t}{s}{f}>"]
      same_move += [f"<{c}{s}{b}>"]
      a = None
      b = None
      c = None
    same_move_list.update({same_move[0]: same_move})
  print(f" same move = {same_move}")
print(f" same move list = {same_move_list}")
print(f" same move list length = {len(same_move_list)}")
file= "list_of_same_move.json"
#if not os.path.isfile(file):
with open(file, "w") as f:
  json.dump(same_move_list, f, indent=4)
