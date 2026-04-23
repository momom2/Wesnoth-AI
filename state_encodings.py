# state_encodings.py
# Feature encoding for Wesnoth game state
# NOTE: All coordinates are 0-indexed (Python convention)
# RENAMED from encodings.py to avoid conflict with Python's standard library

import torch
import torch.nn as nn
from typing import List, Set, Tuple
from classes import (
    Unit, Attack, Map, Position, GameState, GlobalInfo,
    UnitAbility, UnitTrait, UnitStatus, Terrain, TerrainModifiers
)
from constants import (
    TERRAIN_EMBEDDING_DIM, UNIT_TYPE_EMBEDDING_DIM, SPECIAL_EMBEDDING_DIM,
    MAX_UNIT_TYPES, MAX_MAP_WIDTH, MAX_MAP_HEIGHT
)

class AttackEncoder(nn.Module):
    """Encodes unit attacks."""
    
    def __init__(self):
        super().__init__()
        self.damage_type_embedder = nn.Embedding(6, 16)
        self.weapon_special_embedder = nn.Embedding(12, SPECIAL_EMBEDDING_DIM)
        
    def encode_attack(self, attack: Attack) -> torch.Tensor:
        """Encode a single attack."""
        damage_type_emb = self.damage_type_embedder(
            torch.tensor(attack.type_id, dtype=torch.long)
        )
        
        # Encode specials (attention over all present specials)
        if attack.weapon_specials:
            special_indices = torch.tensor(
                [s.value for s in attack.weapon_specials],
                dtype=torch.long
            )
            special_embs = self.weapon_special_embedder(special_indices)
            specials_encoding = special_embs.mean(dim=0)
        else:
            specials_encoding = torch.zeros(SPECIAL_EMBEDDING_DIM)
        
        # Numerical features (normalized)
        numerical = torch.tensor([
            attack.number_strikes / 6.0,
            attack.damage_per_strike / 40.0,
            float(attack.is_ranged)
        ])
        
        return torch.cat([damage_type_emb, specials_encoding, numerical])

class UnitEncoder(nn.Module):
    """Encodes units."""
    
    def __init__(self, max_unit_types: int = MAX_UNIT_TYPES):
        super().__init__()
        self.unit_type_embedder = nn.Embedding(max_unit_types, UNIT_TYPE_EMBEDDING_DIM)
        self.ability_embedder = nn.Embedding(14, SPECIAL_EMBEDDING_DIM)
        self.trait_embedder = nn.Embedding(12, SPECIAL_EMBEDDING_DIM)
        self.status_embedder = nn.Embedding(4, SPECIAL_EMBEDDING_DIM)
        self.attack_encoder = AttackEncoder()
        
        # Learnable queries for attention
        self.ability_query = nn.Parameter(torch.randn(SPECIAL_EMBEDDING_DIM))
        self.trait_query = nn.Parameter(torch.randn(SPECIAL_EMBEDDING_DIM))
        self.status_query = nn.Parameter(torch.randn(SPECIAL_EMBEDDING_DIM))
    
    def encode_abilities_traits_statuses(
        self,
        abilities: Set[UnitAbility],
        traits: Set[UnitTrait],
        statuses: Set[UnitStatus]
    ) -> torch.Tensor:
        """Encode abilities, traits, and statuses using attention."""
        
        # Abilities
        if abilities:
            ability_indices = torch.tensor([a.value for a in abilities], dtype=torch.long)
            ability_embs = self.ability_embedder(ability_indices)
            scores = torch.matmul(ability_embs, self.ability_query)
            weights = torch.softmax(scores, dim=0)
            abilities_enc = torch.matmul(weights, ability_embs)
        else:
            abilities_enc = torch.zeros(SPECIAL_EMBEDDING_DIM)
        
        # Traits
        if traits:
            trait_indices = torch.tensor([t.value for t in traits], dtype=torch.long)
            trait_embs = self.trait_embedder(trait_indices)
            scores = torch.matmul(trait_embs, self.trait_query)
            weights = torch.softmax(scores, dim=0)
            traits_enc = torch.matmul(weights, trait_embs)
        else:
            traits_enc = torch.zeros(SPECIAL_EMBEDDING_DIM)
        
        # Statuses
        if statuses:
            status_indices = torch.tensor([s.value for s in statuses], dtype=torch.long)
            status_embs = self.status_embedder(status_indices)
            scores = torch.matmul(status_embs, self.status_query)
            weights = torch.softmax(scores, dim=0)
            statuses_enc = torch.matmul(weights, status_embs)
        else:
            statuses_enc = torch.zeros(SPECIAL_EMBEDDING_DIM)
        
        return torch.cat([abilities_enc, traits_enc, statuses_enc])
    
    def encode_unit(self, unit: Unit) -> torch.Tensor:
        """Encode complete unit."""
        # Type embedding
        unit_type_emb = self.unit_type_embedder(torch.tensor(unit.name_id, dtype=torch.long))
        
        # Numerical features (normalized)
        numerical = torch.tensor([
            unit.max_hp / 100.0,
            unit.current_hp / 100.0,
            unit.max_exp / 150.0,
            unit.current_exp / 150.0,
            unit.max_moves / 10.0,
            unit.current_moves / 10.0,
            unit.cost / 100.0,
            unit.side / 2.0,
            unit.alignment.value / 3.0,
            float(unit.is_leader),
            float(unit.has_attacked)
        ])
        
        # Special features
        special_features = self.encode_abilities_traits_statuses(
            unit.abilities, unit.traits, unit.statuses
        )
        
        # Resistances and defenses
        resistances = torch.tensor(unit.resistances, dtype=torch.float32)
        defenses = torch.tensor(unit.defenses, dtype=torch.float32)
        
        # Movement costs (normalized)
        movement_costs = torch.tensor(unit.movement_costs, dtype=torch.float32) / 10.0
        
        return torch.cat([
            unit_type_emb,
            numerical,
            special_features,
            resistances,
            defenses,
            movement_costs
        ])

class GameStateEncoder(nn.Module):
    """Encodes complete game state."""
    
    def __init__(self):
        super().__init__()
        self.terrain_embedder = nn.Embedding(14, TERRAIN_EMBEDDING_DIM)
        self.unit_encoder = UnitEncoder()
        self.device = None  # Will be set when moved to device
    
    def to(self, device):
        """Override to track device."""
        self.device = device
        return super().to(device)
    
    def encode_hex(
        self,
        terrain_types: Set[Terrain],
        modifiers: Set[TerrainModifiers],
        unit: Unit = None,
        map_width: int = 50,
        map_height: int = 50,
        position: Position = None
    ) -> torch.Tensor:
        """Encode a single hex."""
        
        # Terrain embedding (average if multiple types)
        if terrain_types:
            terrain_indices = torch.tensor([t.value for t in terrain_types], dtype=torch.long)
            terrain_embs = self.terrain_embedder(terrain_indices)
            terrain_emb = terrain_embs.mean(dim=0)
        else:
            terrain_emb = torch.zeros(TERRAIN_EMBEDDING_DIM)
        
        # Position encoding (normalized, 0-indexed)
        if position:
            pos_enc = torch.tensor([position.x / map_width, position.y / map_height])
        else:
            pos_enc = torch.zeros(2)
        
        # Unit encoding
        if unit:
            unit_enc = self.unit_encoder.encode_unit(unit)
        else:
            # Get the expected unit encoding size dynamically
            dummy_encoding = torch.zeros(
                UNIT_TYPE_EMBEDDING_DIM + 11 + 3*SPECIAL_EMBEDDING_DIM + 6 + 16 + 16
            )
            unit_enc = dummy_encoding
        
        # Modifier flags
        modifier_flags = torch.tensor([
            float(TerrainModifiers.VILLAGE in modifiers),
            float(TerrainModifiers.KEEP in modifiers),
            float(TerrainModifiers.CASTLE in modifiers)
        ])
        
        return torch.cat([terrain_emb, pos_enc, unit_enc, modifier_flags])
    
    def encode_game_state(self, game_state: GameState) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode complete game state.
        
        NOTE: All coordinates are 0-indexed in the internal representation.
        Returns CPU tensors that should be moved to device by caller.
        
        Returns:
            map_representation: [height, width, features]
            global_features: [global_feature_dim]
            fog_mask: [height, width] (1 = visible, 0 = fogged)
        """
        map_obj = game_state.map
        width = min(map_obj.size_x, MAX_MAP_WIDTH)
        height = min(map_obj.size_y, MAX_MAP_HEIGHT)
        
        # Initialize map representation (CPU tensors)
        hex_feature_size = self.get_hex_feature_size()
        map_representation = torch.zeros(height, width, hex_feature_size)
        fog_mask = torch.zeros(height, width)
        
        # Create position -> hex mapping (coordinates are 0-indexed)
        hex_by_pos = {(hex_obj.position.x, hex_obj.position.y): hex_obj for hex_obj in map_obj.hexes}
        
        # Create position -> unit mapping
        unit_by_pos = {(unit.position.x, unit.position.y): unit for unit in map_obj.units}
        
        # Temporarily move encoder to CPU for encoding
        original_device = next(self.parameters()).device
        if original_device.type != 'cpu':
            self.cpu()
        
        # Encode each hex (using 0-indexed coordinates)
        for y in range(height):
            for x in range(width):
                pos = Position(x, y)
                
                # Check if in mask (off-board)
                if any(m.x == x and m.y == y for m in map_obj.mask):
                    continue
                
                # Check if fogged
                is_fogged = any(f.x == x and f.y == y for f in map_obj.fog)
                fog_mask[y, x] = 0.0 if is_fogged else 1.0
                
                # Get hex and unit
                hex_obj = hex_by_pos.get((x, y))
                unit = unit_by_pos.get((x, y))
                
                if hex_obj:
                    hex_enc = self.encode_hex(
                        hex_obj.terrain_types,
                        hex_obj.modifiers,
                        unit,
                        width,
                        height,
                        pos
                    )
                    map_representation[y, x] = hex_enc
        
        # Move encoder back to original device
        if original_device.type != 'cpu':
            self.to(original_device)
        
        # Encode global features
        global_info = game_state.global_info
        current_side_info = game_state.sides[global_info.current_side - 1]
        
        global_features = torch.tensor([
            global_info.current_side / 2.0,
            global_info.turn_number / 50.0,
            current_side_info.current_gold / 500.0,
            global_info.village_gold / 5.0,
            global_info.village_upkeep / 2.0,
            global_info.base_income / 10.0,
            current_side_info.nb_villages_controlled / 20.0
        ])
        
        return map_representation, global_features, fog_mask
    
    def get_hex_feature_size(self) -> int:
        """Calculate hex feature size."""
        return (
            TERRAIN_EMBEDDING_DIM +  # 16
            2 +                       # position
            UNIT_TYPE_EMBEDDING_DIM + # 32
            11 +                      # unit numerical
            3 * SPECIAL_EMBEDDING_DIM + # 48
            6 +                       # resistances
            16 +                      # defenses
            16 +                      # movement costs
            3                         # modifier flags
        )  # Total: 150
