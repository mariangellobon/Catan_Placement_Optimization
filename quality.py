"""
Quality function computation for Catan settlement evaluation.

Computes three components:
1. Resource score: Measures resource type coverage/diversity
2. Expected cards: Expected number of cards per turn based on dice probabilities
3. Probability at least one: Probability of getting at least one resource in a turn
"""

from typing import Set, List
from collections import Counter


def resource_score(vertices: List[int], board) -> float:
    """
    Measure the mix/coverage of resource types the settlements touch.
    
    Args:
        vertices: List of vertex indices where settlements are placed
        board: Board object with tile information
        
    Returns:
        Resource diversity score (higher is better)
    """
    if not vertices:
        return 0.0
    
    # Collect all resource types from tiles adjacent to these vertices
    resource_types = set()
    for vertex in vertices:
        for tile_idx in board.tiles_touching[vertex]:
            tile = board.tiles[tile_idx]
            if tile['resource'] != 'desert':  # Desert doesn't produce resources
                resource_types.add(tile['resource'])
    
    # Count distinct resource types (diversity)
    num_types = len(resource_types)
    
    # Also consider total number of resource-producing tiles
    total_tiles = sum(
        1 for vertex in vertices
        for tile_idx in board.tiles_touching[vertex]
        if board.tiles[tile_idx]['resource'] != 'desert'
    )
    
    # Weighted combination: diversity + coverage
    # Diversity is more important, but total coverage matters too
    return num_types * 2.0 + total_tiles * 0.5


def expected_cards(vertices: List[int], board) -> float:
    """
    Expected number of cards per turn based on dice roll probabilities.
    
    For each tile adjacent to a settlement, with number n:
    P(roll n) * number_of_resources_from_that_tile
    
    Sum over all settlements.
    
    Args:
        vertices: List of vertex indices where settlements are placed
        board: Board object with dice probabilities and tile information
        
    Returns:
        Expected number of cards per turn
    """
    if not vertices:
        return 0.0
    
    expected = 0.0
    
    # Track tiles we've already counted (to avoid double-counting if two vertices touch same tile)
    counted_tiles = set()
    
    for vertex in vertices:
        for tile_idx in board.tiles_touching[vertex]:
            if tile_idx in counted_tiles:
                continue
            
            tile = board.tiles[tile_idx]
            if tile['resource'] == 'desert':
                continue
            
            number_token = tile['number']
            if number_token is None:
                continue
            
            # Probability of rolling this number
            prob = board.dice_probabilities.get(number_token, 0.0)
            
            # Each tile produces 1 resource when its number is rolled
            expected += prob * 1.0
            
            counted_tiles.add(tile_idx)
    
    return expected


def prob_at_least_one(vertices: List[int], board) -> float:
    """
    Probability of getting at least one resource card in a turn.
    
    This counts multiple tiles in the same roll only once.
    Uses inclusion-exclusion principle or complement: 1 - P(no resources)
    
    Args:
        vertices: List of vertex indices where settlements are placed
        board: Board object with dice probabilities and tile information
        
    Returns:
        Probability of getting at least one resource in a turn
    """
    if not vertices:
        return 0.0
    
    # Collect all tiles and their numbers
    tile_numbers = []
    for vertex in vertices:
        for tile_idx in board.tiles_touching[vertex]:
            tile = board.tiles[tile_idx]
            if tile['resource'] == 'desert':
                continue
            number_token = tile['number']
            if number_token is None:
                continue
            tile_numbers.append(number_token)
    
    if not tile_numbers:
        return 0.0
    
    # Use complement: P(at least one) = 1 - P(none)
    # P(none) = product over all possible rolls of P(not getting resource on that roll)
    # But we need to be careful: if multiple tiles have the same number,
    # rolling that number gives resources from all of them
    
    # Group by number token
    number_counts = Counter(tile_numbers)
    
    # For each possible roll (2-12), compute probability of NOT getting any resource
    prob_no_resource = 0.0
    
    for roll in range(2, 13):
        prob_roll = board.dice_probabilities.get(roll, 0.0)
        if roll not in number_counts:
            # This roll doesn't give us any resources, so it contributes to "no resource"
            prob_no_resource += prob_roll
    
    # The complement gives us probability of getting at least one resource
    prob_at_least_one = 1.0 - prob_no_resource
    
    return prob_at_least_one


def compute_quality(vertices: List[int], board, 
                   w_resources: float = 1/3,
                   w_expected_cards: float = 1/3,
                   w_prob_at_least_one: float = 1/3) -> float:
    """
    Combined quality score for a set of settlements.
    
    Args:
        vertices: List of vertex indices where settlements are placed
        board: Board object
        w_resources: Weight for resource score component
        w_expected_cards: Weight for expected cards component
        w_prob_at_least_one: Weight for probability at least one component
        
    Returns:
        Combined quality score
    """
    res_score = resource_score(vertices, board)
    exp_cards = expected_cards(vertices, board)
    prob_one = prob_at_least_one(vertices, board)
    
    benefit = (w_resources * res_score +
               w_expected_cards * exp_cards +
               w_prob_at_least_one * prob_one)
    
    return benefit

