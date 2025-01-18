# assumptions.py

# All the assumptions here need to be checked eventually. Mark them as checked when done.
# Also includes arbitrary choices when they need to be done

MAX_UNIT_TYPE = 1000 # Using this for now. Can't be more than a couple hundred in default 1v1? TODO: Check; if possible, implement a dynamic value for arbitrarily long lists (looking at you, ageless)
MAX_ATTACKS = 4 # We assume no unit has more than 4 attack options. TODO: check this assumption.

SPECIAL_EMBEDDING_DIM = 16 # ARBITRARY. TODO: Sanity check.
UNIT_EMBEDDING_DIM = 32 # ARBITRARY. TODO: Sanity check.
UNIT_ENCODING_DIM = 129 # TODO: Hope I got it right, check more rigorously than by hand.
# UNIT_EMBEDDING_DIM for unit type features
# + 11 for numerical features
# + 3 * SPECIAL_EMBEDDING_DIM for special features
# + 6 for resistance features
# + 16 for defence features
# + 16 for movement cost features
# = 129

MAX_RECRUITS = 20 # We assume the recruitment list is never longer than 20. This would need to be changed to handle arbitrary long lists if we wanted to do campaign. 

MAX_ACTIONS_ALLOWED = 2000 # If the game takes too long, it's better to cut it short than waste time.
# Sanity check: 50 turns, with 20 units moving 2 times each turn for the AI's side. Reasonable upper limit imo.

ACTION_PENALTY = 0.01 # Small penalty for each action to encourage action-efficiency
TIMEOUT_PENALTY = 1.0 # Penalty for reaching MAX_ACTIONS_ALLOWED

# Memory-related constants
MEMORY_STATE_SIZE = 256  # Size of the memory state vector for each game
REPLAY_BUFFER_SIZE = 100000  # Number of experiences to keep in memory
REPLAY_BATCH_SIZE = 512  # Size of training batches
CHECKPOINT_FREQUENCY = 1000  # Save model every N games
REPLAY_SAVE_FREQUENCY = 100  # Save detailed replay every N games