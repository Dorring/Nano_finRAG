from src.evaluation.nf_eval_04 import rank_bucket


def test_bm25_and_dense_ranks_are_bucketed_separately():
    assert rank_bucket(1) == "top20"
    assert rank_bucket(20) == "top20"
    assert rank_bucket(21) == "21_40"
    assert rank_bucket(40) == "21_40"
    assert rank_bucket(41) == "41_100"
    assert rank_bucket(100) == "41_100"
    assert rank_bucket(101) == "101_200"
    assert rank_bucket(200) == "101_200"
    assert rank_bucket(None) == "not_retrieved"
