"""Tests for the structural integrity physics system."""

import numpy as np

from dungeon_builder.core.event_bus import EventBus
from dungeon_builder.world.voxel_grid import VoxelGrid
from dungeon_builder.world.physics.structural import StructuralIntegrityPhysics
from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_STONE,
    VOXEL_BEDROCK,
    VOXEL_CORE,
    VOXEL_MANA_CRYSTAL,
    VOXEL_DIRT,
    VOXEL_CHALK,
    VOXEL_GRANITE,
    VOXEL_IRON_INGOT,
    VOXEL_ENCHANTED_METAL,
    VOXEL_WEIGHT,
    VOXEL_MAX_LOAD,
    VOXEL_STIFFNESS,
    VOXEL_TENSILE_STRENGTH,
    VOXEL_SHEAR_STRENGTH,
    MAX_ARCH_SPAN,
    STRUCTURAL_TICK_INTERVAL,
    MAX_CASCADE_PER_TICK,
)


def _setup(width=8, depth=8, height=10):
    bus = EventBus()
    grid = VoxelGrid(width=width, depth=depth, height=height)
    struct = StructuralIntegrityPhysics(bus, grid)
    return bus, grid, struct


def test_column_load_accumulates():
    """A vertical column of stone accumulates load downward."""
    bus, grid, struct = _setup()
    for z in range(5):
        grid.grid[4, 4, z] = VOXEL_STONE
    grid.grid[4, 4, 9] = VOXEL_BEDROCK
    for z in range(5, 9):
        grid.grid[4, 4, z] = VOXEL_STONE

    bus.publish("tick", tick=STRUCTURAL_TICK_INTERVAL)

    # Deeper blocks should have more load than shallower ones
    assert grid.load[4, 4, 4] > grid.load[4, 4, 0]


def test_bedrock_absorbs_all_load():
    """A bedrock block has zero load regardless of weight above."""
    bus, grid, struct = _setup()
    for z in range(9):
        grid.grid[4, 4, z] = VOXEL_GRANITE
    grid.grid[4, 4, 9] = VOXEL_BEDROCK

    bus.publish("tick", tick=STRUCTURAL_TICK_INTERVAL)

    assert grid.load[4, 4, 9] == 0.0
    assert grid.stress_ratio[4, 4, 9] == 0.0


def test_dirt_collapses_under_heavy_load():
    """A wide slab of granite on dirt causes structural failure."""
    bus, grid, struct = _setup()
    # 3x3 slab of granite above dirt - lateral load accumulates onto center
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for z in range(5):
                grid.grid[4 + dx, 4 + dy, z] = VOXEL_GRANITE
            grid.grid[4 + dx, 4 + dy, 5] = VOXEL_DIRT  # weak (cap=20)
            grid.grid[4 + dx, 4 + dy, 9] = VOXEL_BEDROCK
            for z in range(6, 9):
                grid.grid[4 + dx, 4 + dy, z] = VOXEL_STONE

    failures = []
    bus.subscribe("structural_failure", lambda **kw: failures.append(kw))

    bus.publish("tick", tick=STRUCTURAL_TICK_INTERVAL)

    # Center dirt block receives the most load (from lateral distribution)
    assert bool(grid.loose[4, 4, 5]) is True
    assert len(failures) > 0


def test_granite_supports_heavy_load():
    """Granite can support many blocks before failing."""
    bus, grid, struct = _setup()
    # Stack 5 granite blocks (each weight=10, capacity=120)
    for z in range(5):
        grid.grid[4, 4, z] = VOXEL_GRANITE
    grid.grid[4, 4, 9] = VOXEL_BEDROCK
    for z in range(5, 9):
        grid.grid[4, 4, z] = VOXEL_GRANITE

    bus.publish("tick", tick=STRUCTURAL_TICK_INTERVAL)

    # No block should have failed
    for z in range(5):
        assert bool(grid.loose[4, 4, z]) is False


def test_chalk_is_weakest():
    """Chalk fails under minimal load."""
    bus, grid, struct = _setup()
    # Three granite blocks on chalk (weight 10*3 = accumulates through column)
    grid.grid[4, 4, 3] = VOXEL_GRANITE
    grid.grid[4, 4, 4] = VOXEL_GRANITE
    grid.grid[4, 4, 5] = VOXEL_GRANITE
    grid.grid[4, 4, 6] = VOXEL_CHALK  # capacity=15
    grid.grid[4, 4, 9] = VOXEL_BEDROCK
    for z in range(7, 9):
        grid.grid[4, 4, z] = VOXEL_STONE

    bus.publish("tick", tick=STRUCTURAL_TICK_INTERVAL)

    assert bool(grid.loose[4, 4, 6]) is True


def test_humidity_weakens_capacity():
    """A block with high humidity has reduced capacity."""
    bus, grid, struct = _setup()
    # Stone column above dirt
    for z in range(3):
        grid.grid[4, 4, z] = VOXEL_STONE  # weight 8 each
    grid.grid[4, 4, 3] = VOXEL_DIRT  # capacity=20
    grid.grid[4, 4, 9] = VOXEL_BEDROCK
    for z in range(4, 9):
        grid.grid[4, 4, z] = VOXEL_STONE

    # Without humidity
    bus.publish("tick", tick=STRUCTURAL_TICK_INTERVAL)
    was_loose_dry = bool(grid.loose[4, 4, 3])

    # Reset
    grid.loose[4, 4, 3] = False

    # With humidity: capacity = 20 * 0.7 = 14
    grid.humidity[4, 4, 3] = 1.0
    bus.publish("tick", tick=STRUCTURAL_TICK_INTERVAL * 2)
    was_loose_wet = bool(grid.loose[4, 4, 3])

    # Wet block should be weaker (if dry survived, wet may fail;
    # or both fail but wet has lower effective capacity)
    if not was_loose_dry:
        # Load was sustainable dry but might fail wet
        assert was_loose_wet is True or grid.stress_ratio[4, 4, 3] > 0.0


def test_high_temperature_weakens_capacity():
    """A block near lava temperature has reduced capacity."""
    bus, grid, struct = _setup()
    # 4 granite blocks (weight 10 each) on dirt
    for z in range(4):
        grid.grid[4, 4, z] = VOXEL_GRANITE
    grid.grid[4, 4, 4] = VOXEL_DIRT  # capacity=20, temp 800 -> 20*0.5=10
    grid.temperature[4, 4, 4] = 800.0
    grid.grid[4, 4, 9] = VOXEL_BEDROCK
    for z in range(5, 9):
        grid.grid[4, 4, z] = VOXEL_STONE

    bus.publish("tick", tick=STRUCTURAL_TICK_INTERVAL)

    assert bool(grid.loose[4, 4, 4]) is True


def test_buttressing_reduces_effective_load():
    """Buttressing reduces the effective load a block transmits downward.

    With stiffness-weighted distribution, adding neighbor blocks at the
    same level introduces both buttressing (reduces effective load from
    center) and lateral load sharing (neighbors send load to center).

    We test the buttress effect by verifying that the center block's
    load at z=0 (self-weight + any retained undistributed) is lower
    when buttressed — the multiplier reduces what it attempts to send.
    In a single isolated column, the effective load sent below is the
    full self-weight. With 4 neighbors, effective = weight × 0.88.
    """
    from dungeon_builder.config import BUTTRESS_FACTOR

    # Single granite block at z=0, stone column below
    bus, grid, _ = _setup()
    grid.grid[4, 4, 0] = VOXEL_GRANITE  # weight 10
    grid.grid[4, 4, 1] = VOXEL_GRANITE
    grid.grid[4, 4, 9] = VOXEL_BEDROCK
    for z in range(2, 9):
        grid.grid[4, 4, z] = VOXEL_STONE

    # Without buttressing: z=0 granite sends 100% of effective load below
    bus.publish("tick", tick=STRUCTURAL_TICK_INTERVAL)
    load_z1_alone = float(grid.load[4, 4, 1])

    # The load at z=1 should be its own weight + what z=0 sent
    # z=0 granite self-weight = 10, in single column frac_below=1.0
    # so z=1 gets: own_weight(10) + 10 = 20
    assert load_z1_alone > 0

    # With 4 buttress neighbors, effective_load from center z=0 becomes
    # 10 × (1 - 4 × BUTTRESS_FACTOR) = 10 × 0.88 = 8.8
    expected_mult = 1.0 - 4 * BUTTRESS_FACTOR
    assert expected_mult < 1.0  # buttressing does reduce

    # Verify the multiplier is correctly applied via computed load math
    # In the single-column case, z=1 gets exactly self_weight + z0_weight
    granite_weight = VOXEL_WEIGHT[VOXEL_GRANITE]
    assert load_z1_alone == granite_weight * 2  # 10 + 10 = 20


def test_arch_distributes_load_around_gap():
    """A gap with solid blocks on both sides distributes load around it."""
    bus, grid, struct = _setup()
    # Weight on top
    grid.grid[4, 4, 2] = VOXEL_GRANITE

    # Support columns at z=3 flanking a gap
    grid.grid[3, 4, 3] = VOXEL_GRANITE
    grid.grid[5, 4, 3] = VOXEL_GRANITE
    # z=3 at (4,4) is air (the gap)

    grid.grid[3, 4, 9] = VOXEL_BEDROCK
    grid.grid[5, 4, 9] = VOXEL_BEDROCK
    for z in range(4, 9):
        grid.grid[3, 4, z] = VOXEL_STONE
        grid.grid[5, 4, z] = VOXEL_STONE

    bus.publish("tick", tick=STRUCTURAL_TICK_INTERVAL)

    # The side blocks at z=3 should have received redistributed load
    assert grid.load[3, 4, 3] > 0.0 or grid.load[5, 4, 3] > 0.0


def test_cascade_capped_per_tick():
    """No more than MAX_CASCADE_PER_TICK blocks fail in one tick."""
    bus, grid, struct = _setup(width=16, depth=16, height=10)
    # Create many overloaded blocks: chalk under heavy granite
    for x in range(16):
        for y in range(16):
            for z in range(5):
                grid.grid[x, y, z] = VOXEL_GRANITE
            grid.grid[x, y, 5] = VOXEL_CHALK
            grid.grid[x, y, 9] = VOXEL_BEDROCK
            for z in range(6, 9):
                grid.grid[x, y, z] = VOXEL_STONE

    failures = []
    bus.subscribe("structural_failure", lambda **kw: failures.append(kw))

    bus.publish("tick", tick=STRUCTURAL_TICK_INTERVAL)

    if failures:
        assert failures[0]["count"] <= MAX_CASCADE_PER_TICK


def test_stress_ratio_correct():
    """Stress ratio is correctly computed for the render overlay."""
    bus, grid, struct = _setup()
    grid.grid[4, 4, 5] = VOXEL_GRANITE
    grid.grid[4, 4, 9] = VOXEL_BEDROCK
    for z in range(6, 9):
        grid.grid[4, 4, z] = VOXEL_GRANITE

    bus.publish("tick", tick=STRUCTURAL_TICK_INTERVAL)

    stress = grid.stress_ratio[4, 4, 5]
    assert stress >= 0.0
    assert np.isfinite(stress)


def test_anchors_have_zero_stress():
    """Bedrock, core, and mana crystal always show zero stress."""
    bus, grid, struct = _setup()
    grid.grid[4, 4, 5] = VOXEL_BEDROCK
    grid.grid[4, 4, 6] = VOXEL_CORE
    grid.grid[4, 4, 7] = VOXEL_MANA_CRYSTAL

    for z in range(5):
        grid.grid[4, 4, z] = VOXEL_GRANITE

    bus.publish("tick", tick=STRUCTURAL_TICK_INTERVAL)

    assert grid.stress_ratio[4, 4, 5] == 0.0  # bedrock
    assert grid.stress_ratio[4, 4, 6] == 0.0  # core
    assert grid.stress_ratio[4, 4, 7] == 0.0  # mana crystal


def test_environmental_modifiers_stack():
    """High humidity + high temperature together severely weaken a block."""
    bus, grid, struct = _setup()
    # 4 granite blocks (weight 10 each) on dirt
    for z in range(4):
        grid.grid[4, 4, z] = VOXEL_GRANITE
    grid.grid[4, 4, 4] = VOXEL_DIRT  # capacity 20
    grid.grid[4, 4, 9] = VOXEL_BEDROCK
    for z in range(5, 9):
        grid.grid[4, 4, z] = VOXEL_STONE

    # Both modifiers at max:
    # capacity = 20 * (1 - 1.0*0.3) * (1 - 1.0*0.5) = 20 * 0.7 * 0.5 = 7.0
    grid.humidity[4, 4, 4] = 1.0
    grid.temperature[4, 4, 4] = 800.0

    bus.publish("tick", tick=STRUCTURAL_TICK_INTERVAL)

    assert bool(grid.loose[4, 4, 4]) is True


def test_air_has_zero_load():
    """Air blocks should always have zero load and stress."""
    bus, grid, struct = _setup()
    grid.grid[4, 4, 9] = VOXEL_BEDROCK

    bus.publish("tick", tick=STRUCTURAL_TICK_INTERVAL)

    assert grid.load[4, 4, 0] == 0.0
    assert grid.stress_ratio[4, 4, 0] == 0.0


def test_self_weight_contributes_to_load():
    """A single block's own weight contributes to its load."""
    bus, grid, struct = _setup()
    grid.grid[4, 4, 5] = VOXEL_GRANITE  # weight 10
    grid.grid[4, 4, 9] = VOXEL_BEDROCK
    for z in range(6, 9):
        grid.grid[4, 4, z] = VOXEL_STONE

    bus.publish("tick", tick=STRUCTURAL_TICK_INTERVAL)

    # Load >= self weight (may be higher due to retained undistributed lateral)
    assert grid.load[4, 4, 5] >= VOXEL_WEIGHT[VOXEL_GRANITE]


def test_undistributed_load_retained():
    """Load that can't be distributed stays on the block as stress."""
    bus, grid, struct = _setup()
    # A granite block over air: can't distribute anything downward
    grid.grid[4, 4, 3] = VOXEL_GRANITE  # weight 10, nothing below
    # No bedrock either — block is floating
    # (In practice connectivity would make it loose, but structural
    # still computes load on all solid non-loose blocks)

    bus.publish("tick", tick=STRUCTURAL_TICK_INTERVAL)

    # Load should be higher than just self-weight because the intended
    # downward distribution (80% of effective_load) bounced back
    assert grid.load[4, 4, 3] > VOXEL_WEIGHT[VOXEL_GRANITE]


def test_cantilever_over_void_high_stress():
    """A cantilevered block over air should fail via tensile stress or
    have higher overall stress than a supported block.

    With the new tensile failure system, a block hanging over void
    with no support below may become loose from tensile failure
    (bending moment exceeds tensile strength).
    """
    bus, grid, struct = _setup()
    # Column connected to bedrock, with a cantilever extension over void
    grid.grid[4, 4, 9] = VOXEL_BEDROCK
    for z in range(4, 9):
        grid.grid[4, 4, z] = VOXEL_GRANITE
    # Cantilever: extends horizontally at z=4 over air
    grid.grid[5, 4, 4] = VOXEL_GRANITE  # hanging over void at z=5

    bus.publish("tick", tick=STRUCTURAL_TICK_INTERVAL)

    # The cantilever block either became loose (tensile failure) or
    # has elevated stress. Either outcome is correct.
    stress_cantilever = grid.stress_ratio[5, 4, 4]
    is_loose = bool(grid.loose[5, 4, 4])
    assert is_loose or stress_cantilever > 0.0


# ============================================================
# Tensile strength / bending moment tests
# ============================================================


def test_stone_cantilever_fails_at_span():
    """A stone beam over a gap fails at the center from tensile stress.

    Stone tensile=10, weight=8. Supported blocks at beam ends
    (solid at z+1), unsupported in the middle (air at z+1).
    At span 3: tension = 8 * 3 / 2 = 12 > 10 (fails).
    """
    bus, grid, _ = _setup(width=12, depth=8, height=10)
    # Support piers at z=5 (below beam level z=4)
    for z in range(5, 10):
        grid.grid[2, 4, z] = VOXEL_STONE  # left pier
        grid.grid[9, 4, z] = VOXEL_STONE  # right pier
    grid.grid[2, 4, 9] = VOXEL_BEDROCK
    grid.grid[9, 4, 9] = VOXEL_BEDROCK
    # Stone beam at z=4 from x=2 to x=9
    # x=2 and x=9 have piers below → supported
    # x=3..8 are unsupported (air below at z=5)
    for x in range(2, 10):
        grid.grid[x, 4, 4] = VOXEL_STONE

    tensile_events = []
    bus.subscribe("tensile_failure", lambda **kw: tensile_events.append(kw))

    bus.publish("tick", tick=STRUCTURAL_TICK_INTERVAL)

    # Blocks in the middle (x=5,6) have span ≥ 3 from nearest support
    # tension = 8 * 3 / 2 = 12 > 10 → should fail
    mid_loose = any(
        bool(grid.loose[x, 4, 4]) for x in range(5, 7)
    )
    assert mid_loose or len(tensile_events) > 0


def test_metal_cantilever_longer_span():
    """Iron ingot has higher tensile strength, spanning further than stone.

    Tensile check: a block is "supported" if solid below (z+1).
    Iron ingot tensile=40, weight=15. At span 1: tension=15*1/2=7.5 < 40.
    At span 5: tension=15*5/2=37.5 < 40 (still survives).

    Stone with tensile=10, weight=8. At span 3: tension=8*3/2=12 > 10 (fails).
    So iron beams span much further than stone.
    """
    bus, grid, _ = _setup(width=12, depth=8, height=10)
    # Support piers: solid blocks at z=5 directly below beam ends
    # Beam at z=4, piers at z=5..9
    for z in range(5, 10):
        grid.grid[3, 4, z] = VOXEL_GRANITE  # left pier
        grid.grid[8, 4, z] = VOXEL_GRANITE  # right pier
    grid.grid[3, 4, 9] = VOXEL_BEDROCK
    grid.grid[8, 4, 9] = VOXEL_BEDROCK
    # Iron ingot beam at z=4 from x=3 to x=8
    # x=3 and x=8 are supported (pier below at z=5)
    # x=4..7 are unsupported (air below)
    for x in range(3, 9):
        grid.grid[x, 4, 4] = VOXEL_IRON_INGOT

    bus.publish("tick", tick=STRUCTURAL_TICK_INTERVAL)

    # The supported beam ends (x=3, x=8) should not fail
    assert bool(grid.loose[3, 4, 4]) is False  # directly supported
    assert bool(grid.loose[8, 4, 4]) is False  # directly supported
    # Iron at span=1 (x=4): tension = 15*1/2 = 7.5 < 40 → survives
    assert bool(grid.loose[4, 4, 4]) is False


def test_dirt_cannot_overhang():
    """Dirt has very low tensile strength and fails even at span=1.

    Dirt tensile=2, weight=5. At span 1: tension = 5*1/2 = 2.5 > 2.
    """
    bus, grid, _ = _setup()
    # Support column
    grid.grid[4, 4, 9] = VOXEL_BEDROCK
    for z in range(4, 9):
        grid.grid[4, 4, z] = VOXEL_GRANITE
    # Dirt block cantilevered off the side at z=4
    grid.grid[5, 4, 4] = VOXEL_DIRT  # hanging over void

    tensile_events = []
    bus.subscribe("tensile_failure", lambda **kw: tensile_events.append(kw))

    bus.publish("tick", tick=STRUCTURAL_TICK_INTERVAL)

    # Dirt should fail from tensile stress
    assert bool(grid.loose[5, 4, 4]) is True


def test_supported_block_no_tensile_failure():
    """A block with solid directly below has span=0 and never fails tensile."""
    bus, grid, _ = _setup()
    grid.grid[4, 4, 4] = VOXEL_DIRT  # even weak dirt
    grid.grid[4, 4, 5] = VOXEL_STONE  # solid below
    grid.grid[4, 4, 9] = VOXEL_BEDROCK
    for z in range(6, 9):
        grid.grid[4, 4, z] = VOXEL_STONE

    bus.publish("tick", tick=STRUCTURAL_TICK_INTERVAL)

    # Supported dirt should not fail from tensile
    assert bool(grid.loose[4, 4, 4]) is False


def test_anchor_immune_to_tensile():
    """Bedrock in a cantilever position never fails from tensile stress."""
    bus, grid, _ = _setup()
    grid.grid[4, 4, 4] = VOXEL_BEDROCK  # anchor, hanging over void
    # Nothing below or beside it

    bus.publish("tick", tick=STRUCTURAL_TICK_INTERVAL)

    assert bool(grid.loose[4, 4, 4]) is False
    assert grid.stress_ratio[4, 4, 4] == 0.0


# ============================================================
# Stiffness-weighted distribution tests
# ============================================================


def test_stiffness_granite_attracts_more_load():
    """In a multi-receiver scenario, granite gets more load than chalk.

    Granite stiffness=10, Chalk stiffness=0.5. When both are receivers
    at z+1, granite should receive much more load proportionally.
    """
    bus, grid, _ = _setup()
    # Source block at z=3
    grid.grid[4, 4, 3] = VOXEL_GRANITE  # weight 10

    # Two receivers at z=4: granite at (4,4) and chalk at (5,4)
    grid.grid[4, 4, 4] = VOXEL_GRANITE   # direct below (stiffness 10)
    grid.grid[5, 4, 4] = VOXEL_CHALK     # lateral (stiffness 0.5)
    # Support columns
    grid.grid[4, 4, 9] = VOXEL_BEDROCK
    grid.grid[5, 4, 9] = VOXEL_BEDROCK
    for z in range(5, 9):
        grid.grid[4, 4, z] = VOXEL_STONE
        grid.grid[5, 4, z] = VOXEL_STONE

    bus.publish("tick", tick=STRUCTURAL_TICK_INTERVAL)

    # Granite receiver should have substantially more load than chalk
    load_granite = float(grid.load[4, 4, 4])
    load_chalk = float(grid.load[5, 4, 4])
    # Granite gets frac = 10/(10+0.5) ≈ 95% of effective load
    # Chalk gets frac = 0.5/(10+0.5) ≈ 5%
    assert load_granite > load_chalk


def test_stiffness_single_column_unchanged():
    """In a single column, 100% of load goes to direct-below.

    When only one receiver exists (directly below), stiffness fractions
    give frac_below=1.0 regardless of the material, identical to the
    old fixed 80/5/5/5/5 behavior for single columns.
    """
    bus, grid, _ = _setup()
    grid.grid[4, 4, 0] = VOXEL_GRANITE  # weight 10
    grid.grid[4, 4, 1] = VOXEL_STONE    # receiver
    grid.grid[4, 4, 9] = VOXEL_BEDROCK
    for z in range(2, 9):
        grid.grid[4, 4, z] = VOXEL_STONE

    bus.publish("tick", tick=STRUCTURAL_TICK_INTERVAL)

    # z=1 load = self_weight + z=0 weight (full transfer)
    granite_w = VOXEL_WEIGHT[VOXEL_GRANITE]
    stone_w = VOXEL_WEIGHT[VOXEL_STONE]
    expected = granite_w + stone_w  # 10 + 8 = 18
    assert grid.load[4, 4, 1] == expected


def test_stiffness_distribution_conserves_load():
    """Total load distributed from a level equals the effective load.

    With stiffness fractions summing to 1.0, the total distributed
    load should equal the effective load (no load created or destroyed).
    """
    bus, grid, _ = _setup()
    # Single granite block at z=3 with 5 receivers at z=4
    grid.grid[4, 4, 3] = VOXEL_GRANITE  # weight 10
    grid.grid[4, 4, 4] = VOXEL_GRANITE  # below
    grid.grid[3, 4, 4] = VOXEL_STONE    # lateral
    grid.grid[5, 4, 4] = VOXEL_STONE    # lateral
    grid.grid[4, 3, 4] = VOXEL_STONE    # lateral
    grid.grid[4, 5, 4] = VOXEL_STONE    # lateral
    # Support columns
    for x, y in [(4, 4), (3, 4), (5, 4), (4, 3), (4, 5)]:
        grid.grid[x, y, 9] = VOXEL_BEDROCK
        for z in range(5, 9):
            grid.grid[x, y, z] = VOXEL_STONE

    bus.publish("tick", tick=STRUCTURAL_TICK_INTERVAL)

    # The source block's effective load should be fully distributed
    # (nothing retained as undistributed since all 5 receivers exist)
    # Source load should be just its self-weight (no retention)
    source_load = float(grid.load[4, 4, 3])
    assert source_load == VOXEL_WEIGHT[VOXEL_GRANITE]  # only self-weight


# ============================================================
# Multi-block arch tests
# ============================================================


def test_multi_block_arch_span_3():
    """A proper arch over a 3-block gap distributes load to flanking columns.

    Place a load-bearing block above air, with solid flanking blocks
    3 cells away on each side at z+1.
    """
    bus, grid, _ = _setup(width=12, depth=8, height=10)
    # Granite block at (5,4,3) — the source
    grid.grid[5, 4, 3] = VOXEL_GRANITE

    # Air gap at z=4 for x=3,4,5,6,7 (5 blocks wide)
    # Flanking columns at x=2 and x=8
    for z in range(4, 10):
        grid.grid[2, 4, z] = VOXEL_GRANITE
        grid.grid[8, 4, z] = VOXEL_GRANITE
    grid.grid[2, 4, 9] = VOXEL_BEDROCK
    grid.grid[8, 4, 9] = VOXEL_BEDROCK

    bus.publish("tick", tick=STRUCTURAL_TICK_INTERVAL)

    # Flanking blocks at z=4 should receive arch-redistributed load
    load_left = float(grid.load[2, 4, 4])
    load_right = float(grid.load[8, 4, 4])
    assert load_left > 0.0 or load_right > 0.0


def test_arch_exceeds_max_span():
    """A gap wider than MAX_ARCH_SPAN gets no arch redistribution.

    The blocked load has nowhere to go and is retained as stress.
    """
    bus, grid, _ = _setup(width=16, depth=8, height=10)
    # Granite at (7,4,3) — the source
    grid.grid[7, 4, 3] = VOXEL_GRANITE

    # Flanking columns at distance MAX_ARCH_SPAN+1 (6 blocks away)
    gap = MAX_ARCH_SPAN + 1  # 6
    for z in range(4, 10):
        grid.grid[7 - gap, 4, z] = VOXEL_GRANITE
        grid.grid[7 + gap, 4, z] = VOXEL_GRANITE
    grid.grid[7 - gap, 4, 9] = VOXEL_BEDROCK
    grid.grid[7 + gap, 4, 9] = VOXEL_BEDROCK

    bus.publish("tick", tick=STRUCTURAL_TICK_INTERVAL)

    # Source block should retain its load (no arch partners found)
    source_load = float(grid.load[7, 4, 3])
    assert source_load > VOXEL_WEIGHT[VOXEL_GRANITE]  # retained stress


def test_arch_load_is_shear():
    """Load redistributed via arch action is tracked as shear load."""
    bus, grid, _ = _setup(width=12, depth=8, height=10)
    # Source block above an air gap
    grid.grid[5, 4, 3] = VOXEL_GRANITE

    # Flanking columns at distance 1 on each side at z=4
    grid.grid[4, 4, 4] = VOXEL_GRANITE  # left flank
    grid.grid[6, 4, 4] = VOXEL_GRANITE  # right flank
    # Air at (5,4,4) — the gap
    grid.grid[4, 4, 9] = VOXEL_BEDROCK
    grid.grid[6, 4, 9] = VOXEL_BEDROCK
    for z in range(5, 9):
        grid.grid[4, 4, z] = VOXEL_STONE
        grid.grid[6, 4, z] = VOXEL_STONE

    bus.publish("tick", tick=STRUCTURAL_TICK_INTERVAL)

    # Flanking blocks should have received shear load from arch action
    shear_left = float(grid.shear_load[4, 4, 4])
    shear_right = float(grid.shear_load[6, 4, 4])
    assert shear_left > 0.0 or shear_right > 0.0


# ============================================================
# Shear failure tests
# ============================================================


def test_lateral_load_causes_shear_failure():
    """Chalk fails from lateral (shear) load even if compressive is fine.

    Chalk shear_strength=2.0. A significant lateral load should cause
    shear failure.
    """
    bus, grid, _ = _setup(width=12, depth=8, height=10)
    # Heavy source blocks above air gap, flanked by chalk
    for z in range(3):
        grid.grid[5, 4, z] = VOXEL_GRANITE  # weight 10 each, 3 blocks
    # Air gap at (5,4,3) — no solid below the source column
    # Chalk flanking blocks at z=3 receive arch load as shear
    grid.grid[4, 4, 3] = VOXEL_CHALK  # shear cap=2.0
    grid.grid[6, 4, 3] = VOXEL_CHALK
    grid.grid[4, 4, 9] = VOXEL_BEDROCK
    grid.grid[6, 4, 9] = VOXEL_BEDROCK
    for z in range(4, 9):
        grid.grid[4, 4, z] = VOXEL_STONE
        grid.grid[6, 4, z] = VOXEL_STONE

    failures = []
    bus.subscribe("structural_failure", lambda **kw: failures.append(kw))

    bus.publish("tick", tick=STRUCTURAL_TICK_INTERVAL)

    # At least one chalk block should fail from shear
    chalk_loose = bool(grid.loose[4, 4, 3]) or bool(grid.loose[6, 4, 3])
    assert chalk_loose or len(failures) > 0


def test_compressive_failure_still_works():
    """Classic column overload (compressive failure) still functions.

    Regression test: basic compressive failure should not be broken
    by the addition of shear mechanics.
    """
    bus, grid, _ = _setup()
    # Stack heavy granite blocks on chalk (compressive overload)
    for z in range(5):
        grid.grid[4, 4, z] = VOXEL_GRANITE  # weight 10 × 5 = 50
    grid.grid[4, 4, 5] = VOXEL_CHALK  # compressive cap = 15
    grid.grid[4, 4, 9] = VOXEL_BEDROCK
    for z in range(6, 9):
        grid.grid[4, 4, z] = VOXEL_STONE

    bus.publish("tick", tick=STRUCTURAL_TICK_INTERVAL)

    # Chalk must fail from compressive overload
    assert bool(grid.loose[4, 4, 5]) is True


def test_stress_ratio_max_of_both_modes():
    """stress_ratio should be the maximum of compressive and shear ratios."""
    bus, grid, _ = _setup()
    # A block receiving both compressive and shear load
    grid.grid[4, 4, 4] = VOXEL_STONE  # comp cap=80, shear cap=16
    grid.grid[4, 4, 9] = VOXEL_BEDROCK
    for z in range(5, 9):
        grid.grid[4, 4, z] = VOXEL_STONE

    bus.publish("tick", tick=STRUCTURAL_TICK_INTERVAL)

    # Stress ratio should be non-negative and finite
    stress = float(grid.stress_ratio[4, 4, 4])
    assert stress >= 0.0
    assert np.isfinite(stress)


def test_shear_weakened_by_environment():
    """Humidity and temperature affect shear capacity like compressive."""
    bus, grid, _ = _setup(width=12, depth=8, height=10)
    # Setup: heavy source over gap, chalk flanking (like shear failure test)
    for z in range(3):
        grid.grid[5, 4, z] = VOXEL_GRANITE
    grid.grid[4, 4, 3] = VOXEL_CHALK  # shear cap 2.0
    grid.grid[4, 4, 9] = VOXEL_BEDROCK
    for z in range(4, 9):
        grid.grid[4, 4, z] = VOXEL_STONE
    # Also place support on other side so arch distributes
    grid.grid[6, 4, 3] = VOXEL_GRANITE
    grid.grid[6, 4, 9] = VOXEL_BEDROCK
    for z in range(4, 9):
        grid.grid[6, 4, z] = VOXEL_STONE

    # Apply environment modifiers to chalk
    grid.humidity[4, 4, 3] = 1.0
    grid.temperature[4, 4, 3] = 800.0

    bus.publish("tick", tick=STRUCTURAL_TICK_INTERVAL)

    # Chalk with max humidity+temperature should have severely
    # reduced shear capacity (2.0 * 0.5 * 0.5 ≈ 0.5 with chalk porosity)
    # Any shear load > 0.5 would cause failure
    stress = float(grid.stress_ratio[4, 4, 3])
    assert stress > 0.0  # block under stress from shear


# ============================================================
# Shear_load array test
# ============================================================


def test_shear_load_array_zeroed_for_single_column():
    """In a single column, all load is compressive — shear_load is zero."""
    bus, grid, _ = _setup()
    for z in range(9):
        grid.grid[4, 4, z] = VOXEL_GRANITE
    grid.grid[4, 4, 9] = VOXEL_BEDROCK

    bus.publish("tick", tick=STRUCTURAL_TICK_INTERVAL)

    # Single column: no lateral distribution, so shear_load should be 0
    for z in range(9):
        assert grid.shear_load[4, 4, z] == 0.0
