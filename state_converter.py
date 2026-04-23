# state_converter.py
# Convert between Wesnoth WML format and internal game state representation
# FIXED: Changed from JSON to WML parsing (wesnoth.format_json doesn't exist)
# NOTE: Wesnoth uses 1-indexed coordinates, Python uses 0-indexed

from typing import Dict, List, Set, Optional
from dataclasses import dataclass
import re

from classes import (
    GameState, Map, Unit, Attack, Position, Hex, GlobalInfo, SideInfo,
    Alignment, UnitAbility, UnitTrait, UnitStatus, DamageType, Terrain,
    TerrainModifiers, AttackSpecial
)

class StateConverter:
    """Converts between Wesnoth WML and internal representation."""
    
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
    
    def parse_wml(self, wml_string: str) -> Dict:
        """
        Parse WML string into a dictionary structure.
        This is a simple parser for the WML format output by wml.tostring().
        """
        result = {}
        current_dict = result
        dict_stack = []
        current_tag = None
        
        lines = wml_string.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for opening tag [tag_name]
            if line.startswith('[') and not line.startswith('[/') and line.endswith(']'):
                tag_name = line[1:-1]
                new_dict = {}
                
                # Store in parent as list if multiple same-named tags exist
                if tag_name in current_dict:
                    if not isinstance(current_dict[tag_name], list):
                        current_dict[tag_name] = [current_dict[tag_name]]
                    current_dict[tag_name].append(new_dict)
                else:
                    current_dict[tag_name] = new_dict
                
                dict_stack.append((current_dict, tag_name))
                current_dict = new_dict
                current_tag = tag_name
                
            # Check for closing tag [/tag_name]
            elif line.startswith('[/') and line.endswith(']'):
                if dict_stack:
                    current_dict, current_tag = dict_stack.pop()
                
            # Parse attribute key=value
            elif '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Try to convert to appropriate type
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                else:
                    try:
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass  # Keep as string
                
                current_dict[key] = value
        
        return result
    
    def convert_wml_to_json_format(self, wml_dict: Dict) -> Dict:
        """
        Convert parsed WML dictionary to the JSON format expected by the rest of the code.
        This bridges the gap between WML format and the original JSON structure.
        """
        # The WML structure should already be in the right format
        # but we need to ensure lists are properly handled
        
        # Helper function to normalize single items to lists where expected
        def ensure_list(data, key):
            """Ensure a key's value is a list."""
            if key in data:
                if not isinstance(data[key], list):
                    data[key] = [data[key]] if data[key] else []
            else:
                data[key] = []
        
        # Normalize map structure
        if 'map' in wml_dict:
            map_data = wml_dict['map']
            ensure_list(map_data, 'hexes')
            ensure_list(map_data, 'units')
            ensure_list(map_data, 'fog')
            ensure_list(map_data, 'mask')
            
            # Handle alternate key names that might come from WML
            if 'hex' in map_data:
                map_data['hexes'] = map_data['hex'] if isinstance(map_data['hex'], list) else [map_data['hex']]
                del map_data['hex']
            if 'unit' in map_data:
                map_data['units'] = map_data['unit'] if isinstance(map_data['unit'], list) else [map_data['unit']]
                del map_data['unit']
        
        # Normalize sides structure
        ensure_list(wml_dict, 'sides')
        if 'side' in wml_dict:
            wml_dict['sides'] = wml_dict['side'] if isinstance(wml_dict['side'], list) else [wml_dict['side']]
            del wml_dict['side']
        
        return wml_dict
    
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
    
    def convert_wml_to_game_state(self, wml_string: str) -> GameState:
        """Convert WML string to GameState object."""
        # Parse WML string to dictionary
        parsed = self.parse_wml(wml_string)
        
        # Convert to JSON-like format expected by rest of code
        data = self.convert_wml_to_json_format(parsed)
        
        # Now convert using the same logic as before
        map_data = data['map']
        
        # Convert hexes
        hexes_data = map_data.get('hexes', [])
        if isinstance(hexes_data, dict):
            hexes_data = [hexes_data]
        hexes = set(self.convert_hex(hex_data) for hex_data in hexes_data)
        
        # Convert units
        units_data = map_data.get('units', [])
        if isinstance(units_data, dict):
            units_data = [units_data]
        units = set(self.convert_unit(unit_data) for unit_data in units_data)
        
        # Convert fog and mask
        fog_data = map_data.get('fog', [])
        if isinstance(fog_data, dict):
            fog_data = [fog_data]
        fog = set(
            Position(x=pos['x'] - 1, y=pos['y'] - 1)
            for pos in fog_data
        )
        
        mask_data = map_data.get('mask', [])
        if isinstance(mask_data, dict):
            mask_data = [mask_data]
        mask = set(
            Position(x=pos['x'] - 1, y=pos['y'] - 1)
            for pos in mask_data
        )
        
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
