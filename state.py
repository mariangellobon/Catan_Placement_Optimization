"""
State management for Catan settlement placement.

Represents the current game state during DFS search, including
settlement placements, feasibility checking, and memoization keys.
"""

from typing import Dict, List, Set, Optional, Hashable
from copy import deepcopy


class State:
    """
    Represents the current game state during DFS.
    
    Tracks which players have placed settlements at which vertices,
    and provides methods for checking feasibility and computing quality.
    """
    
    def __init__(self, board):
        """
        Initialize an empty state.
        
        Args:
            board: Board object (immutable reference)
        """
        self.board = board
        self.houses: Dict[int, List[int]] = {}  # houses[player] = list of vertices
        self.occupied: Dict[int, Optional[int]] = {}  # occupied[vertex] = player or None
        self.available_vertices: Set[int] = set(board.vertices)
        
        # Initialize for all players
        for player in range(1, 5):
            self.houses[player] = []
        
        # Initialize occupied dict
        for vertex in board.vertices:
            self.occupied[vertex] = None
    
    def clone(self) -> "State":
        """
        Create a deep copy of this state.
        
        Returns:
            New State object with copied data
        """
        new_state = State(self.board)
        new_state.houses = deepcopy(self.houses)
        new_state.occupied = deepcopy(self.occupied)
        new_state.available_vertices = self.available_vertices.copy()
        return new_state
    
    def place_settlement(self, player: int, vertex: int) -> None:
        """
        Place a settlement for the given player at the given vertex.
        
        Updates houses, occupied, and available_vertices.
        
        Args:
            player: Player ID (1-4)
            vertex: Vertex ID where settlement is placed
        """
        if vertex not in self.available_vertices:
            raise ValueError(f"Vertex {vertex} is not available")
        
        if not self.is_feasible(player, vertex):
            raise ValueError(f"Placing settlement at vertex {vertex} is not feasible")
        
        self.houses[player].append(vertex)
        self.occupied[vertex] = player
        self.available_vertices.remove(vertex)
    
    def is_feasible(self, player: int, vertex: int) -> bool:
        """
        Check if placing a settlement at vertex is feasible for player.
        
        Rules:
        1. Vertex must be available (not occupied)
        2. Vertex must not be adjacent to any other settlement (distance rule)
        
        Args:
            player: Player ID (1-4)
            vertex: Vertex ID to check
            
        Returns:
            True if placement is feasible, False otherwise
        """
        # Check if vertex is available
        if vertex not in self.available_vertices:
            return False
        
        # Check if vertex is already occupied
        if self.occupied[vertex] is not None:
            return False
        
        # Check distance rule: vertex must not be adjacent to any occupied vertex
        neighbors = self.board.vertex_neighbors.get(vertex, set())
        for neighbor in neighbors:
            if self.occupied[neighbor] is not None:
                return False
        
        return True
    
    def get_feasible_positions(self, player: int) -> List[int]:
        """
        Get all feasible vertex positions for the given player.
        
        Args:
            player: Player ID (1-4)
            
        Returns:
            List of feasible vertex IDs
        """
        feasible = []
        for vertex in self.available_vertices:
            if self.is_feasible(player, vertex):
                feasible.append(vertex)
        return feasible
    
    def pair_quality(self, player: int, v1: int, v2: int) -> float:
        """
        Get precomputed quality score for player with settlements at v1 and v2.
        
        Args:
            player: Player ID (1-4)
            v1: First vertex
            v2: Second vertex
            
        Returns:
            Quality score
        """
        return self.board.pair_quality[player][v1][v2]
    
    def quality_of_player(self, player: int) -> float:
        """
        Get the quality score for a player's current settlements.
        
        Assumes the player has exactly 2 settlements.
        
        Args:
            player: Player ID (1-4)
            
        Returns:
            Quality score for player's two settlements
        """
        houses = self.houses[player]
        if len(houses) != 2:
            raise ValueError(f"Player {player} does not have exactly 2 settlements")
        
        v1, v2 = houses[0], houses[1]
        return self.pair_quality(player, v1, v2)
    
    def make_key(self) -> Hashable:
        """
        Generate a canonical hashable key for memoization.
        
        Returns:
            Tuple representing the state (placements, available vertices)
        """
        # Create sorted list of (player, vertex) pairs
        placements = []
        for p in sorted(self.houses.keys()):
            for v in sorted(self.houses[p]):
                placements.append((p, v))
        
        # Create sorted tuple of available vertices
        available_tuple = tuple(sorted(self.available_vertices))
        
        return (tuple(placements), available_tuple)
    
    def upper_bound_for_player_given_first(self, player: int, first_pos: int) -> float:
        """
        Compute upper bound on quality for player given first settlement at first_pos.
        
        This is the maximum quality achievable if player places first settlement
        at first_pos, assuming all currently feasible second positions remain available
        (ignoring that future players may take some).
        
        Args:
            player: Player ID (1-4)
            first_pos: Vertex ID for first settlement
            
        Returns:
            Upper bound on quality score
        """
        # Get feasible positions for second settlement
        candidates = self.get_feasible_positions(player)
        candidates = [v for v in candidates if v != first_pos]
        
        if not candidates:
            return -float('inf')
        
        # Find maximum pair quality over all feasible second positions
        max_quality = -float('inf')
        for second_pos in candidates:
            quality = self.pair_quality(player, first_pos, second_pos)
            if quality > max_quality:
                max_quality = quality
        
        return max_quality

