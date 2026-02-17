# Dungeon Builder Project Memory

## Architecture
- Panda3D + Python 3.11+ voxel dungeon-builder
- NumPy uint8 3D grid (64x64x21) + float32 arrays for humidity, temperature, load, shear_load, stress_ratio, thermal_fatigue + uint8 water_level, block_state + bool for loose, claimed, visible
- Chunk-based rendering (16x16x1), event bus (pub/sub), tick-based simulation (20 TPS)
- Simulation-rendering separation: all game logic testable without Panda3D window

## Voxel Types (config.py)
- 0-4: Air, Dirt, Stone, Bedrock, Core
- 10-13: Sedimentary (Sandstone, Limestone, Shale, Chalk)
- 20-22: Metamorphic (Slate, Marble, Gneiss)
- 30-32: Igneous (Granite, Basalt, Obsidian)
- 40-43: Ores (Iron, Copper, Gold, Mana Crystal)
- 50-51: Lava, Water
- 60-63: Ingots (Iron, Copper, Gold, Enchanted Metal)
- 70-77: Functional blocks (Reinforced Wall, Spike, Door, Treasure, Rolling Stone, Tarp, Slope, Stairs)
- 78-87: Advanced blocks (Gold Bait, Heat Beacon, Pressure Plate, Iron Bars, Floodgate, Alarm Bell, Fragile Floor, Pipe, Pump, Steam Vent)

## Metal Type System
- **`metal_type` uint8 array** on VoxelGrid: per-voxel metal specification
- Constants: METAL_NONE=0, METAL_IRON=1, METAL_COPPER=2, METAL_GOLD=3
- **ENCHANTED_OFFSET=128**: bit 7 encodes enchantment (immune to melting)
- Helpers: `is_enchanted_metal(mt)`, `base_metal_of(mt)`, `make_enchanted(mt)`
- **Property LUTs**: METAL_MELT_TEMPERATURE, METAL_STRENGTH_MULT, METAL_GREED_APPEAL, METAL_COLORS, METAL_CONDUCTIVITY_MULT
- Compositional: `effective_shear = base_shear[vtype] × METAL_STRENGTH_MULT[base_metal]`
- HELD_TO_METAL mapping for crafting transfer; ORE_TO_METAL for smelting
- METALLIC_BLOCKS, MELTABLE_BLOCKS frozensets for physics checks

### Which blocks use metal_type
- All metallic functional blocks: Reinforced Wall, Spike, Door, Iron Bars, Floodgate, Pressure Plate, Alarm Bell, Pipe, Pump, Heat Beacon, Gold Bait
- Ingots: VOXEL_IRON_INGOT, VOXEL_COPPER_INGOT, VOXEL_GOLD_INGOT, VOXEL_ENCHANTED_METAL

### Metal Melting (temperature.py)
- Blocks in MELTABLE_BLOCKS melt when temp >= METAL_MELT_TEMPERATURE[base_metal]
- Enchanted blocks (metal_type & 128) are immune
- Gold melts at 600°C, copper at 800°C, iron at 1200°C
- Melted blocks become VOXEL_AIR; "metal_melted" event published

### Advanced Block Behaviors
| Block | Crafting Input | Key Behavior |
|-------|---------------|--------------|
| Gold Bait (78) | Enchanted Gold ingot | Looks like treasure to normal vision; arcane sight sees true type; triggers traps on grab |
| Heat Beacon (79) | Copper ingot | Fixed temp source (500°C); damages non-fire-immune intruders |
| Pressure Plate (80) | Enchanted ingot | Step triggers adjacent spikes/doors/floodgates within range 1 |
| Iron Bars (81) | Enchanted ingot | LOS-transparent, impassable; semi-transparent rendering (α=0.7) |
| Floodgate (82) | Enchanted ingot | State-dependent: open=passable+transparent, closed=blocks water; pressure burst opens |
| Alarm Bell (83) | Enchanted ingot | Detects intruders within range 2; cooldown 40 ticks; shares zone info |
| Fragile Floor (84) | Chalk | Looks like stone; collapses after FRAGILE_FLOOR_WEIGHT_THRESHOLD steps; flyers immune |
| Pipe (85) | Base metal ingot | Forms networks; passive heat/humidity conduction weighted by METAL_CONDUCTIVITY_MULT |
| Pump (86) | Base metal ingot (on pipe) | Active directional pumping; block_state=0-5 for direction; left-click cycles |
| Steam Vent (87) | Obsidian (above lava) | Pulses heat+humidity upward through STEAM_VENT_RANGE=3 air cells |

### Pipe & Pump Physics (physics/pipe.py)
- PipePhysics: event-driven, runs every PUMP_TICK_INTERVAL=5 ticks
- Network BFS: 6-connected pipes/pumps form cached networks (invalidated on voxel change)
- Passive conduction: heat/humidity averages across network × PIPE_CONDUCTIVITY_BASE × METAL_CONDUCTIVITY_MULT
- Active pumping: pumps pull from intake (opposite of direction), distribute to network
- Copper pipes conduct best (1.0), then gold (0.9), then iron (0.8)

### Water + Floodgate Physics (water.py)
- Open floodgates (block_state=0) allow water flow (downward + lateral)
- Gate/door pressure burst: closed floodgates/doors under water pressure exceeding effective_shear × WATER_BURST_FACTOR → forced open
- Effective shear = base_shear × METAL_STRENGTH_MULT[base_metal] (gold gates burst easiest)

## Visibility Systems

### Asymmetric Layer Visibility (`rendering/layer_slice.py`)
- Above focus (toward surface, lower z): 2 layers at α=0.15, 0.08
- Below focus (deeper, higher z): 5 layers at α=0.7, 0.5, 0.35, 0.2, 0.1
- Focus layer always fully opaque (α=1.0)
- Config: `LAYER_ALPHA_ABOVE`, `LAYER_ALPHA_BELOW`, `LAYER_MAX_VISIBLE_ABOVE/BELOW`

### Claimed Territory (`world/claimed_territory.py`)
- 6-connected flood-fill from core through VOXEL_AIR and VOXEL_WATER
- Lava blocks propagation (acts as natural barrier)
- Iterative NumPy boolean dilation (same pattern as gravity connectivity)
- `claimed[x,y,z]` bool: True if air/water voxel reachable from core
- `visible[x,y,z]` bool: True if solid block 6-adjacent to claimed territory (water cells not visible — traversable, not walls)
- Core block always visible
- Recomputes every CLAIMED_TICK_INTERVAL (10) ticks
- Publishes `"claimed_territory_changed"` event when territory changes
- Constructor accepts `core_x, core_y, core_z` params (defaults to config CORE_X/Y/Z)
- **Ore X-ray dilation**: After visibility computation, iterates PLAYER_XRAY_RANGE steps of
  6-connected dilation through solid blocks (not air/water). Only ore/crystal blocks
  (XRAY_VISIBLE_TYPES) in the dilated zone become visible. PLAYER_XRAY_RANGE=3, mutable for
  future spells/upgrades. XRAY_VISIBLE_TYPES = {iron_ore, copper_ore, gold_ore, mana_crystal}.

### Fog of War (`rendering/voxel_renderer.py`)
- First check in `_get_color()`: non-visible blocks return FOG_COLOR (0.03, 0.03, 0.04, 1.0)
- Exception: pending digs show dim gold even through fog (t=0.25 blend with gold)
- Applies to ALL render modes (matter, structural, humidity, heat)
- Renderer subscribes to `"claimed_territory_changed"` to rebuild dirty chunks
- **Ingredient highlights**: cyan tint (t=0.5) on matching blocks, auto-clears after 60 ticks (3s)

### Block Texture Noise (`rendering/voxel_renderer.py`)
- Deterministic per-vertex color noise via FNV-1a hash of (gx, gy, gz, face_idx, vert_idx)
- `_vertex_noise()` returns [-1, +1]; applied as additive perturbation to RGB (R×amp, G×amp×0.8, B×amp×0.6 for warm/cool variation)
- Per-material amplitude: VOXEL_NOISE dict (marble=0.12 high/veined, obsidian=0.03 low/glassy), default VERTEX_NOISE_AMPLITUDE=0.07
- Only active in RENDER_MODE_MATTER; structural/humidity/heat modes stay flat
- Fog-of-war blocks stay flat (noise gated on `base_color != FOG_COLOR`)
- Zero external assets, no UV coordinates, no vertex format change
- All overlays (dig gold, door alpha, loose dim) work naturally with noise on top

### Book of Crafting (`building/crafting_journal.py` + `ui/crafting_book_panel.py`)
- **CraftingJournal**: Tracks which recipes player has crafted (testable, no Panda3D)
  - Subscribes to `"craft_success"` events, records unique recipe names
  - On first discovery: publishes `"recipe_discovered"` event with `recipe=name, total=count`
  - `get_all_recipes_display()` returns list of `{name, description, discovered}` dicts in CraftingBook order
  - Undiscovered recipes show `name="???"`, `description="[Craft this recipe to reveal]"`
  - `discover_all()` debug helper marks everything discovered
- **CraftingBookPanel**: Panda3D DirectGui toggle panel
  - Toggle with B key or "Book [B]" HUD button (publishes `"toggle_crafting_book"` event)
  - Discovered recipes: green name + description; Undiscovered: gray "???"
  - Live-updates if recipe discovered while panel is open
  - 15 recipe rows, y_step=0.08, dark semi-transparent background
- **HUD integration**: "Book [B]" button on bottom bar + green "Recipe discovered!" notification via `_show_error()`
- **Recipe pinning**: Click recipe name → toggles pin (underlined name). Pinned recipe shows in floating overlay
  when panel closed. Click overlay to unpin. `_pinned_recipe: str | None` state.
- **Ingredient highlighting**: Click "Requires: ..." label → scans current z-level for visible instances.
  Publishes `"ingredient_highlight"` (cyan tint, 3s auto-clear) or error message if none found.
  Requires `game_state` param in `CraftingBookPanel.__init__()` (passed from main.py).

## QoL & Dig System Overhaul

### Dig System (`building/build_system.py`)
- **Three dig lists**: `pending_digs` (invisible/fog), `dig_queue` (validated, waiting), `active_digs` (progressing)
- **Pending digs**: Blocks not yet visible go to pending — skip ALL validation except duplicate check.
  Promoted to dig_queue when `claimed_territory_changed` or `dig_complete` reveals them.
  Invalid blocks (air, non-diggable, loose) cancelled with `"dig_cancelled"` event when revealed.
- **Cancel dig**: Clicking a block being dug cancels it (pending, queued, or active). Block fully restored.
  `cancel_dig(x,y,z)` searches all three lists. Publishes `"dig_cancelled"`.
- **Click-to-toggle**: `_on_voxel_left_clicked` checks `is_being_dug()` first — if yes, cancels.
- **Multiple concurrent digs**: `MAX_CONCURRENT_DIGS=999` (effectively unlimited).
- **Structural integrity 3×**: All VOXEL_MAX_LOAD, VOXEL_TENSILE_STRENGTH, VOXEL_SHEAR_STRENGTH
  values multiplied by 3 (except inf and 0). Cave-ins now require deliberate action.

### Auto-Switch (`rendering/camera.py`)
- `_on_left_click()` auto-determines mode from block state: loose → move, solid → dig
- No more manual X-key toggle needed (though still available)
- Clicking air on invisible blocks dispatches as "dig" mode (pending dig)

### Ore X-ray (`world/claimed_territory.py`)
- `PLAYER_XRAY_RANGE = 3` (mutable via config, future spells/upgrades)
- `XRAY_VISIBLE_TYPES = frozenset({IRON_ORE, COPPER_ORE, GOLD_ORE, MANA_CRYSTAL})`
- 6-connected dilation from visible blocks through solid (not air/water), N iterations
- Only ore/crystal types in dilated zone become visible

### Gotchas (QoL)
- Dig tests must set `grid.visible[:] = True` to test visible-block validation path (otherwise goes to pending)
- Territory restriction tests updated: invisible blocks → pending (not rejected)
- Structural/impact/thermal/floodgate tests updated for 3× strength values
- `CraftingBookPanel` now takes optional `game_state` param (None-safe for tests)
- Pending digs use duration=0 as placeholder; real duration set on promotion
- `_check_pending_digs()` called on both `claimed_territory_changed` AND `dig_complete` events

## Underworlder Intruder System

### Underworld Archetypes (5 in `intruders/archetypes.py`)
- **Magmawraith**: hp=80, fire_immune, lava traversal, heats adjacent blocks (100°C every 15 ticks), never_retreats, bash_door
- **Boremite**: hp=25, fast digger (1/3 duration vs Tunneler 1/2), swarm unit, never_retreats, loyalty=0.1
- **Stoneskin Brute**: hp=200 (highest), damage=20, 1/4 spike damage, 1/2 rolling stone damage, frenzy at 30%, overseer role
- **Tremorstalker**: arcane_sight=4, darkvision=5, spike_detect=3, lockpick, scout/explorer, one of few UW that retreats
- **Corrosive Crawler**: can_dig + post-dig structural damage (stress_ratio += 0.5 to 6 adjacent solid blocks), never_retreats
- Exports: `UNDERWORLD_ARCHETYPES`, `UNDERWORLD_ARCHETYPE_BY_NAME`

### Underworld Party Templates (4 in `intruders/party.py`)
| Template | Weight | Composition | Size |
|----------|--------|-------------|------|
| Underworld Horde | 0.30 | 4-7 Boremite + 1-2 Crawler + 0-1 Stalker | 5-10 |
| Overseer & Slaves | 0.25 | 1 Brute + 2-4 Boremite + 0-1 Crawler | 3-6 |
| Infernal Vanguard | 0.20 | 2-3 Magmawraith + 0-1 Brute + 1 Stalker | 3-5 |
| Solitary Hunter | 0.25 | 1 (Brute/Magmawraith/Crawler) | 1 |

### Spawning (`intruders/decision.py`)
- Separate timer: UNDERWORLD_SPAWN_INTERVAL=600 (30s), MAX_UNDERWORLD_PARTIES=2, MAX_UNDERWORLDERS_TOTAL=16
- Spawn at random z in [10, 18] on map edges; air preferred, falls back to carving solid
- Caps independent from surface intruder caps
- `is_underworlder` flag on `Intruder` (agent.py) — default False, set True for underworlders

### Retreat
- Underworlders retreat to deep map edges (z ≥ UNDERWORLD_SPAWN_Z_MIN=10), not surface
- `_is_at_deep_edge()`: True if at map edge with z ≥ 10
- Escape when reaching deep edge while RETREATING
- No exit found → revert to ADVANCING (fight to death)
- `_repath_intruder()` uses `_find_underworld_retreat_goal()` for UW retreaters

### Unique Behaviors
- **Magmawraith heat**: `_tick_magmawraith_heat()` every MAGMAWRAITH_HEAT_INTERVAL(15) ticks, 100°C to 6 neighbors
- **Boremite dig**: `_start_digging()` checks archetype name, uses BOREMITE_DIG_DIVISOR(3) for 1/3 duration
- **Corrosive post-dig**: `_apply_corrosive_damage()` adds CORROSIVE_DAMAGE_FACTOR(0.5) to stress_ratio of 6 adjacent solid blocks
- **Stoneskin Brute interactions**: SPIKE_DAMAGE // 4, ROLLING_STONE_DAMAGE // 2 (name checks in interactions.py)

### HUD
- Separate "Underworlders: N" label (green-tint) below intruder count
- `_on_intruder_spawned/removed` routes to correct counter via `is_underworlder` flag
- Archetype breakdown includes underworld archetypes automatically

## Social Dynamics System

### Intruder Level & Status (`intruders/agent.py`, `intruders/archetypes.py`)
- **IntruderStatus** enum: GRUNT (rank 0), VETERAN (rank 1), ELITE (rank 2), CHAMPION (rank 3)
- **STATUS_TRUST**: {GRUNT: 0.5, VETERAN: 1.0, ELITE: 1.5, CHAMPION: 2.0} — how much faction trusts returned intel
- **Level** (1-5): weighted random from LEVEL_WEIGHTS (0.40, 0.30, 0.20, 0.08, 0.02)
- **Level → Status**: 1-2 → GRUNT, 3 → VETERAN, 4 → ELITE, 5 → CHAMPION
- **Stat scaling**: HP *= 1 + (level-1) × 0.15; Damage *= 1 + (level-1) × 0.10
- **Deadly dungeons** shift level distribution toward higher levels via `_reputation.get_level_shift()`
- Status-first leader election: `key=(-status.value, -loyalty, id)`

### Knowledge Archive (`intruders/knowledge_archive.py`)
- **FactionMap**: per-cell `seen[(x,y,z)] → (vtype, tick, trust)`, `hazards`, `treasures`, `doors`, `uncertainty[(x,y,z)] → float 0.0-1.0`
- **KnowledgeArchive**: separate surface and underworld FactionMaps
- **archive_survivor(intruder, tick)**: escaped intruder's PersonalMap merged into faction archive
  - Contradiction (different vtype for same cell): uncertainty += BASE × (archive_trust / (archive_trust + survivor_trust))
  - Confirmation (same vtype): uncertainty *= CONFIRM_DECAY (0.5)
  - Dead intruders are NOT archived (no call to archive_survivor)
- **inject_knowledge(pmap, is_uw, tick, cunning)**: new intruder receives filtered faction knowledge
  - Staleness filter: skip data older than KNOWLEDGE_STALE_TICKS (4000)
  - Uncertainty filter: skip cells above KNOWLEDGE_UNCERTAIN_THRESHOLD (0.7)
  - Cunning adjustment: high cunning → lower threshold → trusts less (threshold - cunning × 0.2)
- **on_voxel_changed(x, y, z)**: player modifications increase uncertainty (+0.4) in both faction archives
  - Subscribed to "voxel_changed" event in decision.py

### Dungeon Reputation (`intruders/reputation.py`)
- **DungeonReputation**: subscribes to "intruder_died", "intruder_escaped", "intruder_collected_treasure"
- **ReputationProfile**: lethality = kills/(kills+escapes), richness = treasure_lost/(kills+treasure_lost)
- **Profiles**: Deadly (lethality > 0.7, richness < 0.4), Rich (richness > 0.4), Unknown (< 5 events)
- **Spawn modifiers**: objective weights, template weights, loyalty modifier, level shift
  - Deadly: +destroy, -pillage, negative loyalty, higher levels
  - Rich: +pillage, -destroy, negative loyalty
  - Unknown: +explore
- Publishes "reputation_changed" event on profile shifts
- HUD shows "Reputation: Unknown/Deadly/Treasure Hoard/Moderate" with colored text

### Morale System (`intruders/agent.py`, `intruders/party.py`, `intruders/decision.py`)
- **morale** float 0.0-1.0, initial = MORALE_BASE (0.7)
- **Sources**:
  - Ally death: -0.15 (MORALE_ALLY_DEATH_PENALTY)
  - Taking damage: -0.03 (MORALE_DAMAGE_PENALTY)
  - New hazard revealed: -0.01 (MORALE_HAZARD_PENALTY)
  - Collecting treasure: +0.1 (MORALE_TREASURE_BONUS)
  - Leader alive: +0.001/tick (MORALE_LEADER_BONUS)
  - Warden aura: +0.002/tick (MORALE_WARDEN_TICK)
  - Natural drift toward MORALE_BASE at 0.001/tick
  - Frenzy override: morale = 1.0
- **Effects**:
  - morale < 0.3: movement slowed × 1.5 (MORALE_SLOW_FACTOR)
  - morale > 0.8: movement sped × 0.8 (MORALE_FAST_FACTOR), damage × 1.2 (MORALE_DAMAGE_BONUS)
  - morale < 0.1: abandon party and flee (MORALE_FLEE_THRESHOLD)
  - morale < 0.3: retreat threshold doubled (MORALE_RETREAT_MULTIPLIER = 2.0)
  - never_retreats overrides all morale-based retreat

### Inter-Faction Encounters (`intruders/decision.py`)
- **TODO: Overhaul faction dynamics** — current implementation is hostile-only; future work should add alliance, avoidance, and negotiation mechanics
- Every FACTION_ENCOUNTER_INTERVAL (5) ticks
- Same-cell different-faction: mutual damage at effective_damage
- Adjacent (6-connected) different-faction: engage only if both ADVANCING/ATTACKING
- Performance guard: skip if no underworlders active
- Publishes "faction_combat" event per engagement

### Territory Restrictions
- **Digging**: only allowed on `visible` blocks (solid blocks adjacent to claimed air = territory border)
- **Pick up**: only allowed on `visible` blocks (solid blocks within territory)
- **Drop on air**: only allowed on `claimed` air cells (within territory)
- **Drop on solid** (crafting): only allowed on `visible` blocks
- **Door toggle**: only allowed on `visible` doors
- All rejections publish `"error_message"` with "Too far from claimed territory"

## Key Patterns
- SeededRNG with `fork("label")` for deterministic subsystem streams
- Physics follow same pattern: LUT construction, tick subscription, vectorized NumPy
- Camera uses DDA ray march for mouse picking
- Camera panning: WASD + Arrow keys (both work), Q/E rotate, T/Y Z-level, mouse wheel zoom
- All physics tick-based, not frame-based

## Gotchas
- np.bool_ vs Python bool: use `bool(val) is True` not `val is True` in tests
- Float32 precision: use `pytest.approx()` for humidity comparisons
- Z coordinate inverted for rendering: world Z = -array_z
- Float precision in TimeManager tests: use slightly over-threshold dt values
- Dig overlay tests must set `grid.visible[:] = True` to avoid fog-of-war interference
- Build/move/crafting tests must set `grid.visible[:] = True` and `grid.claimed[:] = True` to bypass territory checks
- ClaimedTerritorySystem uses constructor params for core position, not config globals (for testability)
- Core block must be placed as VOXEL_CORE in the grid after carving (main.py does `voxel_grid.grid[CORE_X, CORE_Y, CORE_Z] = VOXEL_CORE`)
- _carve_initial_dungeon carves 5×5 room at core level AND z-1 (headroom) so core's top face renders

## File Structure
- `dungeon_builder/config.py` — all constants, colors, porosity, dig durations, layer alphas, fog color
- `dungeon_builder/world/` — voxel_grid, geology, pathfinding, room_detection, claimed_territory, physics/
- `dungeon_builder/building/` — build_system, move_system, crafting_system, crafting_book, crafting_journal
- `dungeon_builder/rendering/` — voxel_renderer, camera, layer_slice, effects, intruder_renderer
- `dungeon_builder/intruders/` — agent, archetypes, decision, interactions, knowledge_archive, party, personal_map, personal_pathfinder, reputation, vision
- `dungeon_builder/dungeon_core/` — core
- `dungeon_builder/ui/` — hud, render_mode_selector, crafting_book_panel
- `dungeon_builder/world/physics/pipe.py` — NEW: Pipe & pump network physics
- `tests/` — 1208 tests across 52 files, all passing
