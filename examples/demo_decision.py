"""One agent decision, end to end (the blueprint's worked example), using only dynamics modules.

    incoming flight -> reachability -> action mask -> pick an action -> outgoing flight

Run from the project root:  PYTHONPATH=src python examples/demo_decision.py
"""
import numpy as np

from badminton_rl.dynamics.action_feasibility import ActionFeasibility, CellGrid, decode_action
from badminton_rl.dynamics.player_motion import PlayerMotionModel, PlayerState
from badminton_rl.dynamics.reachability import ContactEnvelope, can_intercept
from badminton_rl.dynamics.shot_execution import ContactState, ShotType
from badminton_rl.dynamics.shuttle_flight import ShuttleFlightModel, ShuttleState, make_launch_state

flight, motion = ShuttleFlightModel(), PlayerMotionModel()
feas = ActionFeasibility(motion=motion)

# 1. the opponent (on +y) hits a clear toward the agent (on -y)
incoming = flight.integrate_to_event(ShuttleState(np.array([0.3, 5.5, 2.5]), np.array([0.0, -17.0, 16.0])))
print(f"incoming lands at {np.round(incoming.landing_xy, 2)} after {incoming.flight_time:.2f}s")

# 2. can the agent get to it?
agent = motion.start_reaction(PlayerState((0.8, -3.8), (0.0, 0.0), side=-1))
reach = can_intercept(agent, incoming, deadline_s=10.0, envelope=ContactEnvelope(), motion=motion)
print(f"reachability: {reach.reason}, contact t={reach.contact_time_s:.2f}s at {np.round(reach.contact_position_xyz, 2)}, "
      f"margin {reach.time_margin_s:.2f}s")
assert reach.feasible, "player misses: the rally would end here"

# 3. which of the 1,350 macro-actions are legal?
contact = ContactState(reach.contact_position_xyz)
mask = feas.valid_action_mask(reach.player_at_contact, contact)
print(f"valid actions: {mask.n_valid}/1350; masked because: {dict(mask.masked_summary())}")

# 4. pick the first valid drop (deterministic here; the policy would choose in the real env)
idx = next(i for i in np.flatnonzero(mask.mask) if decode_action(int(i))[0] == ShotType.DROP)
shot, landing, recovery = decode_action(int(idx))
launch = mask.launches[(int(shot), landing)]
target = CellGrid().landing_centre(landing, side=-1)
print(f"chose {shot.name} -> landing cell {landing} {np.round(target, 2)}, recovery cell {recovery}; "
      f"speed {launch.speed:.1f} m/s, elevation {launch.elevation_deg:.1f} deg, net margin {launch.margin_to_net:.2f} m")

# 5. outgoing flight and the agent's recovery movement
out = flight.integrate_to_event(ShuttleState(launch.initial_position_xyz, launch.initial_velocity_xyz))
rec_target = CellGrid().recovery_centre(recovery, side=-1)
agent_after = motion.advance(reach.player_at_contact, rec_target, out.flight_time, arrive=True)
print(f"outgoing lands at {np.round(out.landing_xy, 2)} (in={out.in_bounds}) after {out.flight_time:.2f}s; "
      f"agent now at {np.round(agent_after.position_xy, 2)} heading to {np.round(rec_target, 2)}")