# Project Status

## ✅ WORKING - Core Training Pipeline

**Last Updated:** 2026-01-29

### What Works

1. **Transformer Model** ✅
   - Complete architecture implemented
   - 6-layer transformer with multi-head attention
   - Encodes hexes, units, global features, and memory
   - Outputs action logits (unit, target, attack, recruit) + value estimate
   - Successfully processes game states

2. **Game Simulation** ✅
   - Python-based Wesnoth simulator (`wesnoth_wrapper.py`)
   - Generates realistic game states
   - Handles units, combat, recruitment
   - Games complete in 5-30 turns

3. **Training Loop** ✅
   - Parallel game management (tested up to 4 games)
   - State collection from multiple games
   - AI decision generation
   - Action transmission
   - Game completion handling
   - Statistics tracking

4. **Data Pipeline** ✅
   - State → AI Input conversion working
   - All data types validated (Unit, Hex, Position, Map)
   - Experience storage functional

### Test Results

```bash
# Single game test (30 seconds)
games_completed: 2
wins: 2
average_game_length: 0.34 turns/action

# Multi-game test (60 seconds, 4 parallel games)
Successfully launched 4 parallel games
All games running simultaneously
```

### Architecture

```
Simulator → Game State (JSON)
    ↓
Game Manager → Convert to AI Input
    ↓
Transformer (256d, 6 layers, 8 heads)
    ├── Hex Encoder
    ├── Unit Encoder
    ├── Global Encoder
    └── Memory Encoder
    ↓
Transformer Layers (6x)
    ↓
Action Heads
    ├── Unit Selection
    ├── Target Selection
    ├── Attack Selection
    ├── Recruit Selection
    └── Value Estimate
```

## 🚧 TODO - Critical Path

### Priority 1: Training Logic
- [ ] Implement policy loss function
- [ ] Implement value loss function
- [ ] Implement training step (backward pass)
- [ ] Add experience replay sampling
- [ ] Test gradient flow

### Priority 2: Action Selection
- [ ] Implement proper action sampling from logits
- [ ] Add move action generation
- [ ] Add attack action generation
- [ ] Add recruit action generation
- [ ] Validate actions are legal

### Priority 3: Reward System
- [ ] Define reward function (win/loss/draw)
- [ ] Add intermediate rewards (optional)
- [ ] Backpropagate rewards through experiences
- [ ] Test reward signal

### Priority 4: Memory System
- [ ] Update memory after each action
- [ ] Persist memory across turns
- [ ] Test memory learning

## 🔄 Later: Real Wesnoth Integration

Currently using simulator (`USE_SIMULATOR = True`).

**To switch to real Wesnoth:**
1. Set `USE_SIMULATOR = False` in `local_game_launcher.py`
2. Install Wesnoth 1.18
3. Test Lua plugin loads correctly
4. Validate state serialization format
5. Test action execution

**Lua Plugin Status:**
- Written but untested
- Needs validation against actual Wesnoth API
- Likely requires debugging

## 📊 Model Specifications

```python
WesnothTransformer:
  d_model: 256
  num_layers: 6
  num_heads: 8
  d_ff: 1024
  dropout: 0.1

  Parameters: ~5-10M (estimate)

  Input:
    - Hexes: variable length
    - Units: variable length
    - Global: 10 features
    - Memory: 256 dimensions

  Output:
    - Unit logits: [num_units]
    - Target logits: [num_hexes]
    - Attack logits: [4]
    - Recruit logits: [20]
    - Value: scalar
```

## 🎯 Next Session Goals

1. **Implement training step** (highest priority)
   - Policy gradient loss
   - Value MSE loss
   - Optimizer step
   - Gradient clipping

2. **Improve action selection**
   - Sample from policy distribution
   - Generate valid move/attack actions
   - Test action diversity

3. **Run overnight training**
   - Start training run
   - Monitor convergence
   - Check for bugs

## 📝 Notes

- All Python code is functional
- Simulator provides adequate training data
- Transformer architecture is complete
- Main blocker: training logic implementation
- System is ready for actual training once losses are implemented

## 🚀 How to Run

```bash
# Test (30 seconds)
python test_single_training_run.py

# Full training (simulator)
python train.py --num-games 4

# Switch to real Wesnoth
# Edit local_game_launcher.py: USE_SIMULATOR = False
python train.py --num-games 1 --wesnoth-exe /path/to/wesnoth
```
