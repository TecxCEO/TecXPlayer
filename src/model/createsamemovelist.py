move_paths=["<rgy>","<rgw>","<rgo>","<rby>","<rbw>","<rbo>","<grw>","<gry>","<grb>","<gow>","<goy>","<gob>","<yrg>","<yrb>","<yrw>","<yog>","<yob>","<yow>"]
mosf={'b':'g','g':'b','o':'r','r':'o','w':'y','y':'w'}
# 1. Three-Character Family Tokens (8 Families * 6 Flips = 48 Tokens)
for fc in move_paths.keys():
  for sc in mosf.keys():
    for tc in mosf.keys():
      if sc not in [fc, mosf[fc]] and tc not in [fc, mosf[fc], sc, mosf[sc]]:
        f=moving_step.strip()[0]
        s=moving_step.strip()[1]
        cc=moving_step.strip()[2]
        
        self._add_token(f"{fc}{sc}{tc}", "Tier_5_Cube", "3_Char_Family_Token")
        self._add_token(f"{fc}{tc}{sc}", "Tier_5_Cube", "3_Char_Family_Token")
        self._add_token(f"{sc}{fc}{tc}", "Tier_5_Cube", "3_Char_Family_Token")
        self._add_token(f"{sc}{tc}{fc}", "Tier_5_Cube", "3_Char_Family_Token")
        self._add_token(f"{tc}{fc}{sc}", "Tier_5_Cube", "3_Char_Family_Token")
        self._add_token(f"{tc}{sc}{fc}", "Tier_5_Cube", "3_Char_Family_Token")
        """
        self._add_token(f"{fc}{sc}{tc}", "Tier_5_Cube", "3_Char_Family_Token")
        self._add_token(f"{fc}{tc}{sc}", "Tier_5_Cube", "3_Char_Family_Token")
        self._add_token(f"{sc}{fc}{tc}", "Tier_5_Cube", "3_Char_Family_Token")
        self._add_token(f"{sc}{tc}{fc}", "Tier_5_Cube", "3_Char_Family_Token")
        self._add_token(f"{tc}{fc}{sc}", "Tier_5_Cube", "3_Char_Family_Token")
        self._add_token(f"{tc}{sc}{fc}", "Tier_5_Cube", "3_Char_Family_Token")
        """
