sequences = []
#while True: 
for seq_id in range(99720):
  sequence = {}
  sequence.update("sequence_id": f"training_sequence_00{seq_id}")
  "context_window_batches" = {}
  for bat_idx in range(18):
    context_window_batches.update({"batch_index": bat_idx})
    "steps" = {}
    for stp_idx in range(20):
      "step_index": stp_idx
      {
        "step_index": 0,
        "start_token": <SOS>,
        "control_token": TASK_FWD,
        "current_state":[],
        "move_token": <rgy>,
        "resulting_state":[],
        "eos_token": <EOS>
          }
    sequence.update("sequence_id": f"training_sequence_00{seq_id}")
    sequences += sequence

{
    "sequence_id": "training_sequence_000",
    "context_window_batches": [
      {
        "batch_index": 0,
        "steps": [
          {
            "step_index": 0,
            "start_token": <SOS>,
            "control_token": TASK_FWD,
            "current_state":[],
            "move_token": <rgy>,
            "resulting_state":[],
            "eo_token": <EOS>
          },
          {
            "step_index": 1,
            "sos_token": 100,
            "control_token": 3,
            "current_state":,
            "move_token": 6,
            "resulting_state":,
            "eos_token": 200
          },
          {
            "step_index": 2,
