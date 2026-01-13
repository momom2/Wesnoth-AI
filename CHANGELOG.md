# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Complete Wesnoth AI integration architecture
- `ai_server.py`: JSON-based server for transformer model
- `action_selector.py`: Converts model outputs to game actions
- `lua_ai_bridge.lua`: Wesnoth Lua candidate action
- `test_scenario.cfg`: Wesnoth test scenario
- `test_all.py`: Comprehensive unit test suite (16/19 passing)
- `INTEGRATION_GUIDE.md`: Detailed setup documentation
- `PROJECT_STATUS.md`: Complete project status report
- `CHANGELOG.md`: This file

### Fixed
- **Critical**: Fixed index calculation bug in `action_selector.py:250`
  - Was using hardcoded max width of 100
  - Now uses actual map dimensions from tensor shape
  - Test `test_select_attack_target` now passes
- **Critical**: Fixed Terrain count mismatch in `classes.py:188`
  - Changed defense validation from 16 to 17 to match Terrain enum
  - Updated comment to reflect actual count
- **Critical**: Fixed hashability issue with Map dataclass
  - Changed `hexes` and `units` from `Set` to `List` in Map class
  - Updated `game_manager.py` to use lists instead of sets
  - Added documentation explaining the performance trade-off
- Fixed `[request_choice]` handling in `server.py`
  - Improved logging and error handling
  - Better response format
- Added missing `GameConfig` dataclass to `classes.py`
- Updated `lua_ai_bridge.lua` to use correct port (15001)

### Changed
- Improved all documentation with current status
- Updated `improvements.md` with completed items
- Enhanced test coverage (15 → 16 passing tests)

## [2026-01-13] - Initial Setup

### Added
- Initial project structure
- `transformer.py`: Neural network architecture
- `classes.py`: Data structures for game state
- `encodings.py`: Feature encoding logic
- `assumptions.py`: Configuration parameters
- `server.py`: WML protocol server
- `game_manager.py`: Training manager (incomplete)
- Basic documentation files

### Notes
- Project created in git worktree: `confident-haslett`
- Initial commit message: "Initial commit. game_manager, batch_manager and server are not validated."
