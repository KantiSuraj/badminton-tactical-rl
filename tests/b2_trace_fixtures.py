"""Fixtures copied from real CoachAI+ traces (B2_HISTORY_FULL / B2_OUTPUT_ROWS dumps).

TRACE_C  - a complete rally: B serve (reset), A Clear, B "Missed shot" (rally ends).
TRACE_A  - the last four strokes (ball_rounds 16-19) of a 19-stroke rally that ends with
           B's Push Shot landing out of bounds (finite, outside the 330-unit half-length).
Only the columns the serializer reads are kept (plus the outcome columns it must ignore).
"""
from __future__ import annotations

# ---- TRACE_C -------------------------------------------------------------
C_STATES = [
    [0, [35.2102920838897, -145.25248914538042], [-57.26052977838137, 214.96179746849788], [0.0, 0.0]],
    [0, [35.2102920838897, -145.25248914538042], [-57.26052977838137, 214.96179746849788], [0.0, 0.0]],
    [10, [57.26052977838137, -214.96179746849788], [-13.363463543355465, 243.26671957969666],
     [-101.36628050462721, -242.19119581952137]],
    [2, [13.363463543355465, -243.26671957969666], [85.0, 55.0], [85.0, -165.0]],
]


def _row(ball_round, player, ploc, opp, receive, type_, hit, landing, movement, **extra):
    row = {
        "match": 1.0, "rally": 2.0, "ball_round": float(ball_round), "player": player,
        "player_location_x": ploc[0], "player_location_y": ploc[1],
        "opponent_location_x": opp[0], "opponent_location_y": opp[1],
        "receive_type": float(receive), "type": float(type_),
        "player_hit_x": hit[0], "player_hit_y": hit[1],
        "landing_x": landing[0], "landing_y": landing[1],
        "player_movement_x": movement[0], "player_movement_y": movement[1],
        # outcome / diagnostic columns that must never reach an event
        "getpoint_player": "None", "constraint": 1.0, "violation": "None",
    }
    row.update(extra)
    return row


C_ROWS = [
    _row(1, "B", (35.2102920838897, -145.25248914538042), (-57.26052977838137, 214.96179746849788),
         0, 10, (0.0, 0.0), (101.36628050462721, 242.19119581952137),
         (13.363463543355465, -243.26671957969666)),
    _row(2, "A", (57.26052977838137, -214.96179746849788), (-13.363463543355465, 243.26671957969666),
         10, 2, (-101.36628050462721, -242.19119581952137), (-85.0, 165.0), (-85.0, -55.0)),
    _row(3, "B", (13.363463543355465, -243.26671957969666), (85.0, 55.0),
         2, 11, (0.0, 0.0), (0.0, 0.0), (0.0, 0.0),
         getpoint_player="CHOU Tien Chen", constraint=0.0, violation="Miss hit"),
]

# ---- TRACE_A (ball_rounds 16..19; earlier states are not needed by the serializer) ----
A_STATES = {
    16: [7, [-85.0, -55.0], [23.94578218460083, 239.13859963417053], [-97.03223138267236, -180.22155213616838]],
    17: [4, [-23.94578218460083, -239.13859963417053], [85.0, 55.0], [100.5, -141.0]],
    18: [7, [-85.0, -55.0], [25.853663086891174, 217.97720074653625], [-85.3162104718835, -211.24960423394424]],
    19: [6, [-25.853663086891174, -217.97720074653625], [85.0, 55.0], [100.5, -141.0]],
}
A_ROWS = [
    _row(16, "A", (-85.0, -55.0), (23.94578218460083, 239.13859963417053), 7, 4,
         (-97.03223138267236, -180.22155213616838), (-100.5, 141.0), (-85.0, -55.0)),
    _row(17, "B", (-23.94578218460083, -239.13859963417053), (85.0, 55.0), 4, 7,
         (100.5, -141.0), (85.3162104718835, 211.24960423394424),
         (-25.853663086891174, -217.97720074653625)),
    _row(18, "A", (-85.0, -55.0), (25.853663086891174, 217.97720074653625), 7, 6,
         (-85.3162104718835, -211.24960423394424), (-100.5, 141.0), (-85.0, -55.0)),
    _row(19, "B", (-25.853663086891174, -217.97720074653625), (85.0, 55.0), 6, 3,
         (100.5, -141.0), (27.562238611280918, 372.98964500427246),
         (11.31038248538971, -246.1091959476471),
         getpoint_player="CHOU Tien Chen", constraint=0.0, violation="Out of bound"),
]


def a_states_list():
    """A history list whose indices 16..19 are real; earlier slots are unused placeholders."""
    states = [None] * 20
    for index, state in A_STATES.items():
        states[index] = state
    return states