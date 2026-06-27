import json
sequences = []
for seq_id in range(277): # 20*18*277 = 99720
  print(f" sequences = {sequences}")
  sequence = {}
  sequence.update({"sequence_id": f"training_sequence_{seq_id}"})
  print(f" sequence = {sequence}")
  context_window_batches = {}
  contxt_win_bates = []
  for bat_idx in range(18):
    contxt_win_bat = {}
    contxt_win_bat.update({"batch_index": bat_idx})
    print(f" contxt_win_bat = {contxt_win_bat}")
    steps = {}
    step_idx = []
    for stp_idx in range(20):
      step = {}
      step = {
        "step_index": stp_idx,
        "start_token": "<SOS>",
        "control_token": "TASK_FWD",
        "current_state":[],
        "move_token": "<rgy>",
        "resulting_state":[],
        "eos_token": "<EOS>"
          }
      print(f" step = {step}")
      step_idx += [step]
      print(f" step_idx = {step_idx}")
    # contxt_win_bat.update({"steps" : step_idx})
    contxt_win_bat.update(steps = step_idx)
    print(f" contxt_win_bat = {contxt_win_bat}")
    contxt_win_bates += [contxt_win_bat]
    print(f" contxt_win_bates = {contxt_win_bates}")
  # sequence.update({"context_window_batches" : contxt_win_bates})
  sequence.update(context_window_batches = contxt_win_bates)
  print(f" sequence = {sequence}")
  sequences += [sequence]
  #print(f" sequences = {sequences}")
  file= "training_data_list.json"
  with open(file, "w") as f:
    json.dump(sequences, f, indent=4)
