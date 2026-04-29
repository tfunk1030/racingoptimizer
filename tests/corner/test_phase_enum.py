from racingoptimizer.corner import CornerPhaseKey, Phase


def test_phase_values():
    assert Phase.BRAKING.value == "braking"
    assert Phase.TRAIL_BRAKE.value == "trail_brake"
    assert Phase.MID_CORNER.value == "mid_corner"
    assert Phase.EXIT.value == "exit"
    assert Phase.STRAIGHT.value == "straight"


def test_phase_order():
    assert list(Phase) == [
        Phase.BRAKING,
        Phase.TRAIL_BRAKE,
        Phase.MID_CORNER,
        Phase.EXIT,
        Phase.STRAIGHT,
    ]


def test_corner_phase_key_hashable_and_named():
    key = CornerPhaseKey("sid", 3, 5, Phase.MID_CORNER)
    assert key.session_id == "sid"
    assert key.lap_index == 3
    assert key.corner_id == 5
    assert key.phase is Phase.MID_CORNER
    assert hash(key) == hash(CornerPhaseKey("sid", 3, 5, Phase.MID_CORNER))
    assert {key: 1}[CornerPhaseKey("sid", 3, 5, Phase.MID_CORNER)] == 1
