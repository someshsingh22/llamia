import math

from rewards.parser import parse_popularity_elo
from rewards.puzzle_reward import compute_score


def test_parser_canonical_form():
    assert parse_popularity_elo("The popularity is 88 and the ELO is 1904") == (88, 1904)


def test_parser_handles_negative_popularity():
    assert parse_popularity_elo("popularity is -23 and the ELO is 2100") == (-23, 2100)


def test_parser_picks_last_match_if_model_rambles():
    text = "earlier I guessed popularity is 50 and the ELO is 1000.\nFinal: The popularity is 88 and the ELO is 1904"
    assert parse_popularity_elo(text) == (88, 1904)


def test_parser_returns_none_on_garbage():
    assert parse_popularity_elo("I don't know") == (None, None)


def test_score_perfect_prediction():
    s = compute_score(
        data_source="ssingh22/llamia-chess-data",
        solution_str="The popularity is 88 and the ELO is 1904",
        ground_truth={"popularity": 88, "elo": 1904},
        extra_info=None,
    )
    assert math.isclose(s, 1.0)


def test_score_format_failure_zeros_reward():
    s = compute_score(
        data_source="x",
        solution_str="I think it's a hard puzzle.",
        ground_truth={"popularity": 88, "elo": 1904},
        extra_info=None,
    )
    assert s == 0.0


def test_score_partial_credit_on_close_predictions():
    # 25 popularity off, 200 ELO off → 0.5 * 0.5 + 0.5 * 0.5 = 0.5
    s = compute_score(
        data_source="x",
        solution_str="The popularity is 63 and the ELO is 1704",
        ground_truth={"popularity": 88, "elo": 1904},
        extra_info=None,
    )
    assert math.isclose(s, 0.5, rel_tol=1e-6)


def test_parser_accepts_missing_article_before_elo():
    assert parse_popularity_elo("popularity is 88 and ELO is 1904") == (88, 1904)


def test_parser_rejects_sentence_break():
    # Period after popularity is a format violation per system prompt; must NOT extract.
    assert parse_popularity_elo("popularity is 88. The ELO is 1904") == (None, None)


def test_score_clamps_far_misses_to_zero():
    s = compute_score(
        data_source="x",
        solution_str="The popularity is -100 and the ELO is 400",
        ground_truth={"popularity": 100, "elo": 3000},
        extra_info=None,
    )
    assert s == 0.0
