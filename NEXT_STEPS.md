# Next Steps Guide

This document outlines the immediate next steps to get your Wesnoth AI fully operational.

## Immediate Actions (Next Session)

### 1. Test Wesnoth Integration (Est: 1-2 hours)

#### Install Lua Dependencies
```bash
# Install LuaSocket for TCP communication
luarocks install luasocket

# Install JSON parser
luarocks install dkjson
# OR
luarocks install luajson
```

#### Set Up Wesnoth Add-on
```bash
# Find your Wesnoth user data directory
# Windows: %USERPROFILE%\.local\share\wesnoth\1.XX\
# Linux: ~/.local/share/wesnoth/1.XX/
# macOS: ~/Library/Application Support/Wesnoth_1.XX/

# Create add-on directory
mkdir -p ~/.local/share/wesnoth/1.XX/data/add-ons/TransformerAI

# Copy Lua bridge
cp lua_ai_bridge.lua ~/.local/share/wesnoth/1.XX/data/add-ons/TransformerAI/

# Copy test scenario (optional)
cp test_scenario.cfg ~/.local/share/wesnoth/1.XX/data/scenarios/
```

#### Test Connection
```bash
# Terminal 1: Start AI server
python start_ai.py --debug

# Terminal 2: Launch Wesnoth
# In Wesnoth:
# 1. Go to Multiplayer → Custom Game
# 2. Load "Transformer AI Test" scenario
# 3. Start game

# Expected result: See debug logs showing connection and state exchange
```

**Success Criteria:**
- [ ] Server logs show "New connection from..."
- [ ] Server logs show "Received WML: ..." or game state
- [ ] Server sends action back
- [ ] Wesnoth AI makes a move

**Common Issues:**
- "Module 'socket' not found" → LuaSocket not installed
- "Module 'json' not found" → JSON library not installed
- "Connection refused" → Server not running or wrong port
- No logs → Check Wesnoth's stderr.txt for Lua errors

---

### 2. Validate Training Loop (Est: 1 hour)

#### Test Checkpoint Saving
```bash
# Start server with checkpoint enabled
python start_ai.py --training

# In another terminal, trigger checkpoint save
# (This will happen automatically after games complete)

# Check checkpoints were created
ls -la training/checkpoints/
```

#### Test Checkpoint Loading
```bash
# Start with latest checkpoint
python start_ai.py --load-checkpoint --debug

# Verify in logs:
# "Loading checkpoint: training/checkpoints/checkpoint_X.pt"
# "Successfully loaded checkpoint..."
```

**Success Criteria:**
- [ ] Checkpoints save to `training/checkpoints/`
- [ ] Latest checkpoint loads on startup
- [ ] Training stats restored
- [ ] Only last 5 checkpoints kept

---

### 3. Run First Training Games (Est: 1-2 hours)

#### Single Game Test
```bash
# Start server in training mode
python start_ai.py --training --debug

# Play one game in Wesnoth
# Observe:
# - Actions selected
# - Game completion
# - Replay buffer updates
```

#### Monitor Training
```bash
# Watch training progress
tail -f ai_server.log

# Look for:
# - "Training step - Value Loss: ..."
# - "Saved checkpoint to ..."
# - Game completion messages
```

**Success Criteria:**
- [ ] Game completes without crashes
- [ ] Experience stored in replay buffer
- [ ] Training step executed
- [ ] Losses printed (value, consistency)
- [ ] Checkpoint saved

**Expected Behavior:**
- Initial loss will be random (untrained model)
- Loss should vary as games complete
- Action selection will be mostly random at first

---

## Short-term Improvements (This Week)

### 4. Fix Remaining Test Skips

#### Re-enable Unit Test
The `test_unit_creation` should now work since we fixed the Terrain count:

```python
# In test_all.py, replace the skipTest with actual test
def test_unit_creation(self):
    """Test Unit dataclass."""
    from classes import UnitStatus
    unit = Unit(
        name="Elvish Fighter",
        side=1,
        is_leader=False,
        position=Position(10, 15),
        max_hp=33,
        max_moves=5,
        max_exp=40,
        cost=14,
        alignment=Alignment.NEUTRAL,
        levelup_names=["Elvish Captain", "Elvish Hero"],
        current_hp=33,
        current_moves=5,
        current_exp=0,
        has_attacked=False,
        attacks=[],
        resistances={dt: 0.0 for dt in DamageType},  # 6 values
        defenses={t: 0.5 for t in Terrain},  # 17 values
        movement_costs={t: 1 for t in Terrain},  # 17 values
        abilities=set(),
        traits={UnitTrait.QUICK, UnitTrait.STRONG},
        statuses=set()
    )
    self.assertEqual(unit.name, "Elvish Fighter")
    self.assertEqual(unit.side, 1)
    self.assertEqual(unit.max_hp, 33)
```

**Test and commit:**
```bash
python test_all.py
git add test_all.py
git commit -m "Re-enabled test_unit_creation after fixing Terrain count"
```

---

### 5. Implement Full Policy Loss

Currently, policy loss is a placeholder. Implement proper action replay:

```python
# In game_manager.py, update _train_step():

def _train_step(self, batch: List[Experience]):
    # ... existing code ...

    # 2. Policy Loss (proper implementation)
    policy_loss = 0.0
    for i, exp in enumerate(batch):
        # Extract actual action taken
        action_type = exp.action.get('action_type', 'end_turn')

        # Calculate cross-entropy loss based on action type
        if action_type == 'move':
            # Loss for unit selection
            unit_idx = exp.action.get('unit_idx', 0)
            start_loss = F.cross_entropy(
                start_logits_list[i],
                torch.tensor([unit_idx])
            )

            # Loss for target selection
            target_idx = exp.action.get('target_idx', 0)
            target_loss = F.cross_entropy(
                target_logits_list[i].flatten().unsqueeze(0),
                torch.tensor([target_idx])
            )

            policy_loss += start_loss + target_loss

        elif action_type == 'recruit':
            # Loss for recruit selection
            recruit_idx = exp.action.get('recruit_idx', 0)
            recruit_loss = F.cross_entropy(
                recruit_logits_list[i],
                torch.tensor([recruit_idx])
            )
            policy_loss += recruit_loss

    policy_loss = policy_loss / len(batch)

    # Update total loss
    total_loss = value_loss + policy_loss + consistency_loss
```

---

### 6. Add Metrics Tracking

Create a simple metrics tracker:

```python
# Create metrics.py

import json
from pathlib import Path
from collections import defaultdict

class MetricsTracker:
    def __init__(self, save_path="training/metrics.jsonl"):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics = defaultdict(list)

    def log(self, step: int, **kwargs):
        """Log metrics for a training step."""
        record = {'step': step, **kwargs}
        self.metrics['all'].append(record)

        # Append to file
        with open(self.save_path, 'a') as f:
            f.write(json.dumps(record) + '\n')

    def get_history(self, metric_name: str):
        """Get history of a specific metric."""
        return [m.get(metric_name) for m in self.metrics['all'] if metric_name in m]

# Use in game_manager.py:
from metrics import MetricsTracker

class TrainingManager:
    def __init__(self, ...):
        # ... existing code ...
        self.metrics = MetricsTracker()

    def _train_step(self, batch):
        # ... training code ...

        # Log metrics
        self.metrics.log(
            step=self.training_stats['games_completed'],
            value_loss=value_loss.item(),
            policy_loss=policy_loss.item(),
            consistency_loss=consistency_loss.item(),
            total_loss=total_loss.item(),
            win_rate=self.training_stats['wins'] / max(self.training_stats['games_completed'], 1),
            avg_game_length=self.training_stats['average_game_length']
        )
```

---

## Medium-term Goals (Next Few Weeks)

### 7. Implement Self-Play

Add opponent AI that also uses the transformer:

```python
# Modify test_scenario.cfg to have AI vs AI
[side]
    side=2
    controller=ai
    # Same AI configuration as side 1
[/side]

# Or implement random opponent in Python
class RandomOpponent:
    def select_action(self, game_state):
        """Select random valid action."""
        # ... implementation ...
```

### 8. Add Hyperparameter Tuning

Create a configuration system:

```python
# configs/default.yaml
model:
  embedding_dim: 256
  num_heads: 8
  num_layers: 6

training:
  temperature: 1.0
  exploration: 0.1
  batch_size: 512
  learning_rate: 0.001

# Load in start_ai.py:
python start_ai.py --config configs/default.yaml
```

### 9. Create Visualization Dashboard

Simple web dashboard for monitoring:

```python
# dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px

# Load metrics
df = pd.read_json('training/metrics.jsonl', lines=True)

# Plot losses
st.line_chart(df[['value_loss', 'policy_loss', 'consistency_loss']])

# Plot win rate
st.line_chart(df['win_rate'])

# Run with: streamlit run dashboard.py
```

---

## Long-term Vision (Months)

### 10. Advanced Features

- [ ] EfficientZero value prefix
- [ ] Off-policy correction
- [ ] ReAnalyze from replays
- [ ] Hierarchical memory
- [ ] Multi-scale time horizons
- [ ] Attention-based updates

### 11. Deployment

- [ ] Model compression
- [ ] GPU optimization
- [ ] Distributed training
- [ ] Online API
- [ ] Wesnoth add-on release

### 12. Evaluation

- [ ] Benchmark against default AI
- [ ] Human player evaluation
- [ ] Different maps/scenarios
- [ ] Strategy analysis
- [ ] Replay analysis tools

---

## Quick Reference Commands

```bash
# Start AI server
python start_ai.py

# Start with options
python start_ai.py --temperature 1.5 --exploration 0.2 --load-checkpoint --debug

# Run tests
python test_all.py

# Check server is running
curl http://localhost:15001 || echo "Server not responding"

# Watch logs
tail -f ai_server.log

# Check checkpoints
ls -lh training/checkpoints/

# Clean up temp files
rm -rf __pycache__ tmpclaude-* *.log
```

---

## Troubleshooting

### Server Won't Start
```bash
# Check if port is in use
netstat -an | grep 15001

# Kill existing process
lsof -ti:15001 | xargs kill -9

# Start with different port
python start_ai.py --port 15002
```

### Wesnoth Can't Connect
```bash
# Check server is listening
python start_ai.py --debug

# Verify Lua files are in correct location
find ~/.local/share/wesnoth -name "lua_ai_bridge.lua"

# Check Wesnoth logs
cat ~/.local/share/wesnoth/1.XX/stderr.txt
```

### Training Not Working
```bash
# Verify experiences are being stored
# Add debug logging in game_manager.py:
print(f"Replay buffer size: {len(self.replay_buffer)}")

# Check if training step is being called
# Should see "Training step - Value Loss: ..." in logs
```

---

## Success Metrics

**Week 1:**
- [x] Architecture complete
- [x] Tests passing (16/19)
- [x] Documentation complete
- [ ] Wesnoth integration tested
- [ ] First game played

**Week 2:**
- [ ] Training loop validated
- [ ] Checkpoints working
- [ ] 10+ games played
- [ ] Loss decreasing

**Week 4:**
- [ ] 100+ games played
- [ ] Win rate improving
- [ ] Self-play working
- [ ] Metrics tracked

**Month 2:**
- [ ] 1000+ games played
- [ ] Beats default AI 30%+ of time
- [ ] Strategic behavior emerging
- [ ] Ready for evaluation

---

## Resources

- **Documentation:** README.md, INTEGRATION_GUIDE.md
- **Status:** PROJECT_STATUS.md, SESSION_COMPLETE.md
- **Code:** All in git, well-commented
- **Help:** Check logs first, then documentation

## Getting Help

1. **Check logs:** `tail -f ai_server.log`
2. **Check Wesnoth logs:** `cat ~/.local/share/wesnoth/*/stderr.txt`
3. **Run tests:** `python test_all.py`
4. **Review docs:** INTEGRATION_GUIDE.md has troubleshooting section

---

**Current Status:** Ready for Wesnoth integration testing
**Next Action:** Install Lua dependencies and test connection
**Estimated Time to First Demo:** 3-5 hours

Good luck! 🚀
