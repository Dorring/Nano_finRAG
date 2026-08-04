from scripts.evaluation.replay_nf_opt_11 import replay
def test_replay_has_four_matching_complete_pair_records():
 data=replay()
 assert len(data["records"]) == 4
 assert all(x["strict_binding_pass"] and x["calculator_status"]=="executed" for x in data["records"])
