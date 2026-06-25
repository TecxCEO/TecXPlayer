move_paths=["<rgy>","<rgw>","<rgo>","<rby>","<rbw>","<rbo>","<grw>","<gry>","<grb>","<gow>","<goy>","<gob>","<yrg>","<yrb>","<yrw>","<yog>","<yob>","<yow>"]
mosf={'b':'g','g':'b','o':'r','r':'o','w':'y','y':'w'}
# 1. Three-Character Family Tokens (8 Families * 6 Flips = 48 Tokens)
for moving_step in move_paths.keys():
  f=moving_step.strip()[1]
  s=moving_step.strip()[2]
  t=moving_step.strip()[3]
  if locals.get(same_move):
    #same_move += f"<{f}{s}{t}>"
    same_move += f"<{t}{s}{mosf[f]}>"
    same_move += f"<{mosf[f]}{s}{mosf[t]}>"
    same_move += f"<{mosf[t]}{s}{f}>"
  elif locals.get(same_move) is not exist:
    same_move = f"<{f}{s}{t}>"
    
    # self._add_token(f"{fc}{sc}{tc}", "Tier_5_Cube", "3_Char_Family_Token")
    # self._add_token(f"{fc}{tc}{sc}", "Tier_5_Cube", "3_Char_Family_Token")
    # self._add_token(f"{sc}{fc}{tc}", "Tier_5_Cube", "3_Char_Family_Token")
    # self._add_token(f"{sc}{tc}{fc}", "Tier_5_Cube", "3_Char_Family_Token")
    # self._add_token(f"{tc}{fc}{sc}", "Tier_5_Cube", "3_Char_Family_Token")
    # self._add_token(f"{tc}{sc}{fc}", "Tier_5_Cube", "3_Char_Family_Token")
    """
    self._add_token(f"{fc}{sc}{tc}", "Tier_5_Cube", "3_Char_Family_Token")
    self._add_token(f"{fc}{tc}{sc}", "Tier_5_Cube", "3_Char_Family_Token")
    self._add_token(f"{sc}{fc}{tc}", "Tier_5_Cube", "3_Char_Family_Token")
    self._add_token(f"{sc}{tc}{fc}", "Tier_5_Cube", "3_Char_Family_Token")
    self._add_token(f"{tc}{fc}{sc}", "Tier_5_Cube", "3_Char_Family_Token")
    self._add_token(f"{tc}{sc}{fc}", "Tier_5_Cube", "3_Char_Family_Token")
        """
