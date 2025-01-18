# encodings.py

import torch
import torch.nn as nn
from typing import List, Set, Tuple, Dict
from classes import Unit, Attack, Map, Position, Memory, UnitAbility, UnitTrait, UnitStatus
from assumptions import MAX_ATTACKS, MAX_UNIT_TYPE, UNIT_ENCODING_DIM, UNIT_EMBEDDING_DIM, SPECIAL_EMBEDDING_DIM

class AttackEncoding:
    """
    Handles encoding of unit attacks into tensor representations.
    Each attack combines damage type, weapon specials, and numerical features.
    """
    def __init__(self, weapon_special_dim: int = 16):
        self.damage_type_embedder = nn.Embedding(
            num_embeddings=6,     # The six damage types in Wesnoth
            embedding_dim=16
        )
        
        self.weapon_special_embedder = nn.Embedding(
            num_embeddings=12,    # Number of AttackSpecial enum values
            embedding_dim=weapon_special_dim
        )
    
    def create_attack_query(self, attack: Attack) -> torch.Tensor:
        """
        Creates a query vector for attention over weapon specials.
        Normalizes numerical values to help with training stability.
        """
        return torch.cat([
            self.damage_type_embedder(attack.type_id),
            torch.tensor([
                attack.number_strikes / 6.0,    # Max 6 strikes? (inferno drake) / only need rough normalization so it's fine
                attack.damage_per_strike / 40.0, # Max 40 damage? (dragonguard) / id.
                float(attack.is_ranged)
            ])
        ])
    
    def attend_to_specials(self, query: torch.Tensor, 
                          special_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Uses attention mechanism to combine weapon special embeddings.
        Returns a weighted combination based on relevance to the query.
        """
        if not special_embeddings:
            return torch.zeros(self.weapon_special_embedder.embedding_dim)
            
        embeddings = torch.stack(special_embeddings)
        attention_scores = torch.matmul(query.unsqueeze(0), embeddings.T)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        return torch.matmul(attention_weights, embeddings)
    
    def encode_attack(self, attack: Attack) -> torch.Tensor:
        """Creates a complete fixed-length encoding for an attack."""
        damage_type = self.damage_type_embedder(attack.type_id)
        
        special_embeddings = [
            self.weapon_special_embedder(special.value) 
            for special in attack.weapon_specials
        ]
        query = self.create_attack_query(attack)
        specials_encoding = self.attend_to_specials(query, special_embeddings)
        
        numerical_features = torch.tensor([
            attack.number_strikes,
            attack.damage_per_strike,
            float(attack.is_ranged)
        ])
        
        return torch.cat([
            damage_type,         # 16 dimensions
            specials_encoding,   # weapon_special_dim dimensions
            numerical_features   # 3 dimensions
        ])

class UnitEncoding(nn.Module):
    """
    Handles encoding of units into tensor representations.
    Uses learnable weighted sums to combine multiple abilities and traits.
    """
    def __init__(self, unit_embedding_dim: int = UNIT_EMBEDDING_DIM, special_embedding_dim: int = SPECIAL_EMBEDDING_DIM):
        super().__init__()
        
        # Embedders for categorical features
        self.unit_embedder = nn.Embedding(
            num_embeddings=MAX_UNIT_TYPE,  # Room for many unit types
            embedding_dim=unit_embedding_dim
        )
        
        # Embedders for abilities, traits and statuses
        self.ability_embedder = nn.Embedding(
            num_embeddings=14,  # Number of UnitAbility enum values
            embedding_dim=special_embedding_dim
        )
        self.trait_embedder = nn.Embedding(
            num_embeddings=16,  # Number of UnitTrait enum values
            embedding_dim=special_embedding_dim
        )
        self.status_embedder = nn.Embedding(
            num_embeddings=4,  # Number of UnitStatus enum values
            embedding_dim=special_embedding_dim
        )
        
        # Learnable query vectors for weighted combination
        self.ability_query = nn.Parameter(torch.randn(special_embedding_dim))
        self.trait_query = nn.Parameter(torch.randn(special_embedding_dim))
        self.status_query = nn.Parameter(torch.randn(special_embedding_dim))
        
        self.attack_encoder = AttackEncoding()
    
    def encode_abilities_traits_and_statuses(self, 
                                  abilities: Set[UnitAbility],
                                  traits: Set[UnitTrait],
                                  statuses: Set[UnitStatus]) -> torch.Tensor:
        """
        Encodes abilities traits and statuses using learnable weighted combinations.
        #TODO: Might want to encode abilities and traits together but statuses apart dues to being temporary?
        """
        # Handle abilities
        ability_embeddings = [
            self.ability_embedder(ability.value) 
            for ability in abilities
        ]
        
        if not ability_embeddings:
            abilities_encoding = torch.zeros(self.ability_embedder.embedding_dim)
        else:
            stacked_abilities = torch.stack(ability_embeddings)
            attention_scores = torch.matmul(stacked_abilities, self.ability_query)
            attention_weights = torch.softmax(attention_scores, dim=0)
            abilities_encoding = torch.matmul(attention_weights, stacked_abilities)
        
        # Handle traits
        trait_embeddings = [
            self.trait_embedder(trait.value) 
            for trait in traits
        ]
        
        if not trait_embeddings:
            traits_encoding = torch.zeros(self.trait_embedder.embedding_dim)
        else:
            stacked_traits = torch.stack(trait_embeddings)
            attention_scores = torch.matmul(stacked_traits, self.trait_query)
            attention_weights = torch.softmax(attention_scores, dim=0)
            traits_encoding = torch.matmul(attention_weights, stacked_traits)

        # Handle statuses
        status_embeddings = [
            self.status_embedder(status.value) 
            for status in statuses
        ]
        
        if not status_embeddings:
            statuses_encoding = torch.zeros(self.status_embedder.embedding_dim)
        else:
            stacked_statuses = torch.stack(status_embeddings)
            attention_scores = torch.matmul(stacked_statuses, self.status_query)
            attention_weights = torch.softmax(attention_scores, dim=0)
            statuses_encoding = torch.matmul(attention_weights, stacked_statuses)
        
        return torch.cat([abilities_encoding, traits_encoding, statuses_encoding])
    
    def encode_unit(self, unit: Unit) -> torch.Tensor:
        """Creates a complete encoding for a unit."""
        # Get unit type embedding
        unit_type_features = self.unit_embedder(unit.name_id)
        
        # Encode numerical features
        numerical_features = torch.tensor([
            unit.max_hp,
            unit.current_hp,
            unit.max_exp,
            unit.current_exp,
            unit.max_moves,
            unit.current_moves,
            unit.cost,
            unit.side,
            unit.alignment.value,
            float(unit.is_leader),
            float(unit.has_attacked)
        ])
        
        # Get special abilities and traits encoding
        special_features = self.encode_abilities_traits_and_statuses(
            unit.abilities, unit.traits, unit.statuses
        )
        
        # Include resistances and defenses
        resistance_features = torch.tensor(unit.resistances)
        defense_features = torch.tensor(unit.defenses)
        movement_cost_features = torch.tensor(unit.movement_costs)
        
        return torch.cat([
            unit_type_features,     # unit_embedding_dim dimensions
            numerical_features,     # 11 dimensions
            special_features,       # 3 * special_embedding_dim dimensions
            resistance_features,    # 6 dimensions
            defense_features,       # 16 dimensions
            movement_cost_features  # 16 dimensions
        ])
    
    def encode_unit_attacks(self, unit: Unit) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes all attacks of a unit, returning both encodings and an attention mask.
        """
        attack_encodings = [
            self.attack_encoder.encode_attack(attack)
            for attack in unit.attacks
        ]
        
        # Create mask for valid attacks
        attention_mask = torch.zeros(MAX_ATTACKS)
        attention_mask[:len(attack_encodings)] = 1
        
        # Pad to fixed size
        while len(attack_encodings) < 4:
            attack_encodings.append(torch.zeros_like(attack_encodings[0]))
            
        return torch.stack(attack_encodings), attention_mask

class GameStateEncoding(nn.Module):
    """
    Handles encoding of the complete game state.
    Creates a spatial representation of the map with unit and terrain information.
    """
    def __init__(self, max_map_size: int = 100):
        super().__init__()
        
        self.unit_encoder = UnitEncoding()
        self.terrain_embedder = nn.Embedding(
            num_embeddings=17,  # Number of Terrain enum values
            embedding_dim=16
        )
    
    def get_hex_feature_size(self) -> int: # TODO: Make this dynamic. Right now, it doesn't really check anything.
        """Calculates the total size of the feature vector for each hex."""
        return (
            16 +  # terrain embedding
            2 +   # normalized position coordinates
            UNIT_ENCODING_DIM +  # unit encoding (if present)
            2     # fog of war information # ??? TODO: AI-generated code, to be validated.
        )
    
    def encode_position(self, pos: Position, map_width: int, map_height: int) -> torch.Tensor:
        """Creates a simple normalized position encoding."""
        return torch.tensor([
            pos.x / map_width,
            pos.y / map_height
        ])
    
    # TODO: Encode memory? Check if it should be done here or in transformer.py
    
    def encode_game_state(self, game_map: Map) -> Tuple[torch.Tensor, torch.Tensor]:
        """Creates a complete encoding of the game state."""
        # Initialize the map representation
        map_representation = torch.zeros(
            game_map.size_y,
            game_map.size_x,
            self.get_hex_feature_size()
        )
        
        # Fill in features for each hex
        for hex in game_map.hexes:
            # Get base terrain embedding
            terrain_embedding = torch.mean(torch.stack([
                self.terrain_embedder(terrain.value)
                for terrain in hex.terrain_types
            ]))
            
            # Get position encoding
            pos_encoding = self.encode_position(
                hex.position, game_map.size_x, game_map.size_y
            )
            
            # Find unit at this position (if any)
            unit = next(
                (u for u in game_map.units 
                 if u.position == hex.position),
                None
            )
            
            # Encode unit if present, otherwise use zero vector
            if unit:
                unit_encoding = self.unit_encoder.encode_unit(unit)
            else:
                unit_encoding = torch.zeros(UNIT_ENCODING_DIM)  # Match unit encoding size
            
            # Encode fog of war
            fog_features = torch.tensor([
                float(hex.position in game_map.fog),
                float(hex.position not in game_map.mask)
            ])
            
            # Combine all features
            hex_features = torch.cat([
                terrain_embedding,
                pos_encoding,
                unit_encoding,
                fog_features
            ])
            
            # Store in the map representation
            map_representation[hex.position.y, hex.position.x] = hex_features
        
        # Create global features tensor
        global_features = torch.tensor([
            game_map.current_side,
            game_map.turn_number,
            game_map.time_of_day.value,
            game_map.current_gold,
            game_map.village_gold,
            game_map.village_upkeep,
            game_map.base_income
        ])
        
        return map_representation, global_features