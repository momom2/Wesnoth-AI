# state_converter.py
# Convert the JSON state payload emitted by the Lua state_collector into
# our internal GameState dataclasses.
# NOTE: Wesnoth uses 1-indexed coordinates; Python uses 0-indexed. The
# conversion happens here and ONLY here — do not sprinkle ±1 around.

import json
from typing import Dict, List, Set, Optional

from classes import (
    GameState, Map, Unit, Attack, Position, Hex, GlobalInfo, SideInfo,
    Alignment, UnitAbility, UnitTrait, UnitStatus, DamageType, Terrain,
    TerrainModifiers, AttackSpecial
)

class StateConverter:
    """Converts the Lua-emitted JSON state payload to GameState objects."""
    
    # Mapping dictionaries for enums
    ALIGNMENT_MAP = {
        'lawful': Alignment.LAWFUL,
        'neutral': Alignment.NEUTRAL,
        'chaotic': Alignment.CHAOTIC,
        'liminal': Alignment.LIMINAL
    }
    
    DAMAGE_TYPE_MAP = {
        'blade': DamageType.SLASH,
        'pierce': DamageType.PIERCE,
        'impact': DamageType.IMPACT,
        'fire': DamageType.FIRE,
        'cold': DamageType.COLD,
        'arcane': DamageType.ARCANE
    }
    
    ABILITY_MAP = {
        'ambush': UnitAbility.AMBUSH,
        'concealment': UnitAbility.CONCEALMENT,
        'cures': UnitAbility.CURES,
        'feeding': UnitAbility.FEEDING,
        'heals +4': UnitAbility.HEALS4,
        'heals +8': UnitAbility.HEALS8,
        'illuminates': UnitAbility.ILLUMINATES,
        'leadership': UnitAbility.LEADERSHIP,
        'nightstalk': UnitAbility.NIGHTSTALK,
        'regenerates': UnitAbility.REGENERATES,
        'skirmisher': UnitAbility.SKIRMISHER,
        'steadfast': UnitAbility.STEADFAST,
        'submerge': UnitAbility.SUBMERGE,
        'teleport': UnitAbility.TELEPORT
    }
    
    TRAIT_MAP = {
        'intelligent': UnitTrait.INTELLIGENT,
        'quick': UnitTrait.QUICK,
        'resilient': UnitTrait.RESILIENT,
        'strong': UnitTrait.STRONG,
        'dextrous': UnitTrait.DEXTROUS,
        'fearless': UnitTrait.FEARLESS,
        'feral': UnitTrait.FERAL,
        'healthy': UnitTrait.HEALTHY,
        'dim': UnitTrait.DIM,
        'slow': UnitTrait.SLOW,
        'undead': UnitTrait.UNDEAD,
        'weak': UnitTrait.WEAK
    }
    
    STATUS_MAP = {
        'poisoned': UnitStatus.POISONED,
        'slowed': UnitStatus.SLOW,
        'petrified': UnitStatus.PETRIFIED,
        'stunned': UnitStatus.STUNNED
    }
    
    ATTACK_SPECIAL_MAP = {
        'backstab': AttackSpecial.BACKSTAB,
        'berserk': AttackSpecial.BERSERK,
        'charge': AttackSpecial.CHARGE,
        'drains': AttackSpecial.DRAIN,
        'firststrike': AttackSpecial.FIRSTSTRIKE,
        'magical': AttackSpecial.MAGICAL,
        'marksman': AttackSpecial.MARKSMAN,
        'plague': AttackSpecial.PLAGUE,
        'poison': AttackSpecial.POISON,
        'slow': AttackSpecial.SLOW
    }
    
    # Terrain parsing (simplified - handles base^overlay format)
    TERRAIN_BASE_MAP = {
        'Aa': Terrain.FROZEN,
        'Gg': Terrain.FLAT,
        'Gs': Terrain.FLAT,
        'Gd': Terrain.FLAT,
        'Hh': Terrain.HILLS,
        'Ha': Terrain.HILLS,
        'Mm': Terrain.MOUNTAINS,
        'Ms': Terrain.MOUNTAINS,
        'Md': Terrain.MOUNTAINS,
        'Ww': Terrain.SHALLOWWATER,
        'Wo': Terrain.DEEPWATER,
        'Ss': Terrain.SWAMP,
        'Ds': Terrain.SAND,
        'Rr': Terrain.FLAT,
        'Re': Terrain.FLAT,
        'Ql': Terrain.CAVE,
        'Xu': Terrain.IMPASSABLE,
        'Uu': Terrain.UNWALKABLE,
    }
    
    def __init__(self):
        # Create unit type to ID mapping (shared across all games)
        self.unit_type_to_id = {}
        self.next_unit_id = 0
    
    def get_unit_type_id(self, unit_type_name: str) -> int:
        """Get or create ID for unit type."""
        if unit_type_name not in self.unit_type_to_id:
            self.unit_type_to_id[unit_type_name] = self.next_unit_id
            self.next_unit_id += 1
        return self.unit_type_to_id[unit_type_name]
    
    def wesnoth_to_python_coords(self, x: int, y: int) -> tuple:
        """Convert Wesnoth 1-indexed coordinates to Python 0-indexed."""
        return (x - 1, y - 1)
    
    def python_to_wesnoth_coords(self, x: int, y: int) -> tuple:
        """Convert Python 0-indexed coordinates to Wesnoth 1-indexed."""
        return (x + 1, y + 1)
    
    def convert_attack(self, attack_data: Dict) -> Attack:
        """Convert attack from parsed data to Attack object."""
        damage_type = self.DAMAGE_TYPE_MAP.get(
            attack_data.get('type', 'blade'),
            DamageType.SLASH
        )
        
        specials = set()
        specials_list = attack_data.get('specials', [])
        if isinstance(specials_list, list):
            for special_name in specials_list:
                special = self.ATTACK_SPECIAL_MAP.get(special_name.lower())
                if special:
                    specials.add(special)
        
        return Attack(
            type_id=damage_type,
            number_strikes=attack_data.get('strikes', 1),
            damage_per_strike=attack_data.get('damage', 1),
            is_ranged=attack_data.get('is_ranged', False),
            weapon_specials=specials
        )
    
    def convert_unit(self, unit_data: Dict) -> Unit:
        """Convert unit from parsed data to Unit object."""
        # Convert coordinates from Wesnoth (1-indexed) to Python (0-indexed)
        x, y = self.wesnoth_to_python_coords(unit_data['x'], unit_data['y'])
        position = Position(x=x, y=y)
        
        # Convert attacks
        attacks = []
        attacks_data = unit_data.get('attacks', [])
        if isinstance(attacks_data, dict):
            attacks_data = [attacks_data]
        for attack in attacks_data:
            attacks.append(self.convert_attack(attack))
        
        # Convert abilities
        abilities = set()
        abilities_list = unit_data.get('abilities', [])
        if isinstance(abilities_list, list):
            for ability_name in abilities_list:
                ability = self.ABILITY_MAP.get(ability_name.lower())
                if ability:
                    abilities.add(ability)
        
        # Convert traits
        traits = set()
        traits_list = unit_data.get('traits', [])
        if isinstance(traits_list, list):
            for trait_name in traits_list:
                trait = self.TRAIT_MAP.get(trait_name.lower())
                if trait:
                    traits.add(trait)
        
        # Convert status
        statuses = set()
        status_list = unit_data.get('statuses', [])
        if isinstance(status_list, list):
            for status_name in status_list:
                status = self.STATUS_MAP.get(status_name.lower())
                if status:
                    statuses.add(status)
        
        # Convert alignment
        alignment = self.ALIGNMENT_MAP.get(
            unit_data.get('alignment', 'neutral').lower(),
            Alignment.NEUTRAL
        )
        
        # Convert defenses
        defense_list = []
        defense_dict = unit_data.get('defenses', {})
        terrain_keys = [
            'castle', 'cave', 'deep_water', 'flat', 'forest', 'frozen',
            'fungus', 'hills', 'mountains', 'reef', 'sand', 'shallow_water',
            'swamp', 'unwalkable', 'village', 'impassable'
        ]
        for terrain in terrain_keys:
            defense_list.append(defense_dict.get(terrain, 100) / 100.0)
        
        # Convert movement costs
        movement_cost_list = []
        movement_dict = unit_data.get('movement_costs', {})
        for terrain in terrain_keys:
            movement_cost_list.append(movement_dict.get(terrain, 99))
        
        return Unit(
            id=unit_data.get('id', ''),
            name=unit_data['name'],
            name_id=self.get_unit_type_id(unit_data['name']),
            side=unit_data['side'],
            is_leader=unit_data.get('is_leader', False),
            position=position,
            max_hp=unit_data['max_hp'],
            max_moves=unit_data['max_moves'],
            max_exp=unit_data['max_exp'],
            cost=unit_data.get('cost', 0),
            alignment=alignment,
            levelup_names=unit_data.get('levelup_names', []),
            current_hp=unit_data['current_hp'],
            current_moves=unit_data['current_moves'],
            current_exp=unit_data['current_exp'],
            has_attacked=unit_data.get('has_attacked', False),
            attacks=attacks,
            resistances=unit_data.get('resistances', [0.0] * 6),
            defenses=defense_list,
            movement_costs=movement_cost_list,
            abilities=abilities,
            traits=traits,
            statuses=statuses
        )
    
    def parse_terrain_code(self, terrain_code: str) -> Set[Terrain]:
        """Parse terrain code and return terrain types."""
        terrains = set()
        
        # Split base and overlay
        if '^' in terrain_code:
            base, overlay = terrain_code.split('^', 1)
        else:
            base, overlay = terrain_code, ''
        
        # Map base terrain
        base_terrain = self.TERRAIN_BASE_MAP.get(base, Terrain.FLAT)
        terrains.add(base_terrain)
        
        # Check for special overlays
        if 'V' in overlay:  # Village
            terrains.add(Terrain.VILLAGE)
        if 'F' in overlay:  # Forest
            terrains.add(Terrain.FOREST)
        if 'K' in overlay or 'C' in base:  # Keep or Castle
            terrains.add(Terrain.CASTLE)
        
        return terrains
    
    def convert_hex(self, hex_data: Dict) -> Hex:
        """Convert hex from parsed data to Hex object."""
        # Convert coordinates
        x, y = self.wesnoth_to_python_coords(hex_data['x'], hex_data['y'])
        position = Position(x=x, y=y)
        
        terrain_code = hex_data.get('full_code', 'Gg')
        terrain_types = self.parse_terrain_code(terrain_code)
        
        # Convert modifiers
        modifiers = set()
        modifiers_list = hex_data.get('modifiers', [])
        if isinstance(modifiers_list, list):
            for mod_name in modifiers_list:
                if mod_name == 'village':
                    modifiers.add(TerrainModifiers.VILLAGE)
                elif mod_name == 'keep':
                    modifiers.add(TerrainModifiers.KEEP)
                elif mod_name == 'castle':
                    modifiers.add(TerrainModifiers.CASTLE)
        
        return Hex(
            position=position,
            terrain_types=terrain_types,
            modifiers=modifiers
        )
    
    def convert_payload_to_game_state(self, payload: str) -> GameState:
        """Parse the JSON state payload emitted by the Lua state_collector
        and return a populated GameState.

        Accepts a JSON string. We used to accept WML and hand-parse it;
        that was replaced when Wesnoth's `wml.tostring` proved too strict
        about table shape and the hand-rolled parser too brittle.
        """
        data = json.loads(payload)

        map_data = data['map']
        hexes = set(self.convert_hex(h) for h in map_data.get('hexes', []))
        units = set(self.convert_unit(u) for u in map_data.get('units', []))
        fog = set(Position(x=pos['x'] - 1, y=pos['y'] - 1)
                  for pos in map_data.get('fog', []))
        mask = set(Position(x=pos['x'] - 1, y=pos['y'] - 1)
                   for pos in map_data.get('mask', []))
        
        # Create map
        game_map = Map(
            size_x=map_data['width'],
            size_y=map_data['height'],
            mask=mask,
            fog=fog,
            hexes=hexes,
            units=units
        )
        
        # Convert global info
        current_side = data['current_side']
        sides_data = data.get('sides', [])
        if isinstance(sides_data, dict):
            sides_data = [sides_data]
        
        current_side_data = sides_data[current_side - 1] if sides_data else {}
        
        global_info = GlobalInfo(
            current_side=current_side,
            turn_number=data['turn_number'],
            time_of_day=data.get('time_of_day', 'morning'),
            village_gold=current_side_data.get('village_gold', 2),
            village_upkeep=current_side_data.get('village_support', 1),
            base_income=current_side_data.get('base_income', 0)
        )
        
        # Convert side info
        sides = []
        for side_data in sides_data:
            recruits = side_data.get('recruits', [])
            if not isinstance(recruits, list):
                recruits = [recruits] if recruits else []
            
            side_info = SideInfo(
                player=f"Side {len(sides) + 1}",
                recruits=recruits,
                current_gold=side_data.get('gold', 0),
                base_income=side_data.get('base_income', 0),
                nb_villages_controlled=side_data.get('num_villages', 0)
            )
            sides.append(side_info)
        
        return GameState(
            game_id=data.get('game_id', 'unknown'),
            map=game_map,
            global_info=global_info,
            sides=sides,
            game_over=data.get('game_over', False),
            winner=data.get('winner', None)
        )
    
    def convert_action_to_json(self, action: Dict) -> Dict:
        """Convert internal action to format for Wesnoth (now used for Lua file)."""
        action_type = action.get('type')
        
        if action_type == 'move':
            # Convert coordinates back to Wesnoth 1-indexed
            start_x, start_y = self.python_to_wesnoth_coords(
                action['start_hex'].x, action['start_hex'].y
            )
            target_x, target_y = self.python_to_wesnoth_coords(
                action['target_hex'].x, action['target_hex'].y
            )
            
            return {
                'type': 'move',
                'start_x': start_x,
                'start_y': start_y,
                'target_x': target_x,
                'target_y': target_y
            }
        
        elif action_type == 'attack':
            start_x, start_y = self.python_to_wesnoth_coords(
                action['start_hex'].x, action['start_hex'].y
            )
            target_x, target_y = self.python_to_wesnoth_coords(
                action['target_hex'].x, action['target_hex'].y
            )
            
            return {
                'type': 'attack',
                'start_x': start_x,
                'start_y': start_y,
                'target_x': target_x,
                'target_y': target_y,
                'weapon_index': action.get('attack_index', 0)
            }
        
        elif action_type == 'recruit':
            target_x, target_y = self.python_to_wesnoth_coords(
                action['target_hex'].x, action['target_hex'].y
            )
            
            return {
                'type': 'recruit',
                'unit_type': action['unit_type'],
                'target_x': target_x,
                'target_y': target_y
            }
        
        elif action_type == 'recall':
            target_x, target_y = self.python_to_wesnoth_coords(
                action['target_hex'].x, action['target_hex'].y
            )
            
            return {
                'type': 'recall',
                'unit_id': action['unit_id'],
                'target_x': target_x,
                'target_y': target_y
            }
        
        elif action_type == 'end_turn':
            return {
                'type': 'end_turn'
            }
        
        else:
            raise ValueError(f"Unknown action type: {action_type}")
