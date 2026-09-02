import math

from voiceauth.matching import CandidateScore, cosine_from_squared_distance, identify


def test_normalized_distance_maps_to_cosine() -> None:
    assert cosine_from_squared_distance(0.0) == 1.0
    assert math.isclose(cosine_from_squared_distance(2.0), 0.0)


def test_identify_selects_best_accepted_candidate() -> None:
    result = identify(
        [
            CandidateScore("a", "A", 0.70),
            CandidateScore("b", "B", 0.82),
            CandidateScore("c", "C", 0.40),
        ],
        threshold=0.65,
    )
    assert result.matched is True
    assert result.identity_id == "b"
    assert result.candidate_count == 3


def test_identify_returns_unknown_when_no_candidate_reaches_threshold() -> None:
    result = identify([CandidateScore("a", "A", 0.5)], threshold=0.65)
    assert result.matched is False
    assert result.identity_id is None
