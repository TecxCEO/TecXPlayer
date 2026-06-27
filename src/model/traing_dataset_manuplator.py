sequences = []
for seq_id in range(277): # 20*18*277 = 99720
  sequence = {}
  sequence.update({"sequence_id": f"training_sequence_{seq_id}"})
  context_window_batches = {}
  cntxt_win_bates = []
  for bat_idx in range(18):
    contxt_win_bat = {}
    contxt_win_bat.update({"batch_index": bat_idx})
    steps = {}
    step_idx = []
    for stp_idx in range(20):
      step = {
        "step_index": stp_idx,
        "start_token": <SOS>,
        "control_token": TASK_FWD,
        "current_state":[],
        "move_token": <rgy>,
        "resulting_state":[],
        "eos_token": <EOS>
          }
      step_idx += step
    cntxt_win_bat.update({steps : step_idx})
  cntxt_win_bates += cntxt_win_bat
  sequence.update(context_window_batches: cntxt_win_bates)
  sequences += sequence
  print(f" sequences = {sequences}")
