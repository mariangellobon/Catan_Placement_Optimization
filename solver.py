"""
DFS solver for optimal Catan settlement placement.

Implements backward induction with pruning and memoization to find
optimal settlement placements for all players.
"""

import time
from typing import Optional, Tuple
from state import State
from board import Board


NUM_PLAYERS = 4


class Solver:
    """
    Solver for optimal settlement placement using DFS with backward induction.
    """
    
    def __init__(self, board: Board, enable_pruning: bool = True):
        """
        Initialize solver with a board.
        
        Args:
            board: Board object
            enable_pruning: If False, disables all pruning for comparison
        """
        self.board = board
        self.memo = {}  # (player, state_key) -> best payoff for this player
        self.enable_pruning = enable_pruning
        
        # Metrics
        self.recursive_calls = 0
        self.feasibility_prunings = 0
        self.upper_bound_prunings = 0
        self.memo_hits = 0
        self.memo_misses = 0
        self.start_time = None
        self.end_time = None
    
    def dfs(self, player: int, state: State) -> Optional[State]:
        """
        Recursive DFS to find optimal settlement placements.
        
        Player is about to choose their first settlement.
        After all later players place their settlements, this player
        chooses their second settlement optimally.
        
        Args:
            player: Current player ID (1-4)
            state: Current game state
            
        Returns:
            Best resulting State after all players have placed optimally,
            or None if no valid placement exists
        """
        # Track recursive calls
        self.recursive_calls += 1
        
        # Base case: all players have placed
        if player > NUM_PLAYERS:
            return state
        
        # Memoization key
        state_key = state.make_key()
        memo_key = (player, state_key)
        
        # Check memo
        if memo_key in self.memo:
            self.memo_hits += 1
            # We know the payoff, but we need to return a state
            # For now, we'll continue to explore (memo can help with pruning)
        else:
            self.memo_misses += 1
        
        # Local lower bound for this player at this node
        best_value = -float('inf')
        best_state_for_player = None
        
        # Get feasible positions for first settlement
        first_candidates = state.get_feasible_positions(player)
        
        # If no feasible positions, return None
        if not first_candidates:
            self.memo[memo_key] = -float('inf')
            return None
        
        # Sort candidates by upper bound (or single quality) in descending order
        # This helps improve LB faster, enabling more pruning
        # We also cache the UB values to avoid recomputing them
        candidate_ubs = {}  # Cache for upper bounds
        
        if self.enable_pruning:
            # Sort by upper bound: best candidates first
            candidates_with_ub = []
            for pos in first_candidates:
                if state.is_feasible(player, pos):
                    ub = state.upper_bound_for_player_given_first(player, pos)
                    candidate_ubs[pos] = ub
                    candidates_with_ub.append((ub, pos))
            
            # Sort by upper bound descending (best first)
            candidates_with_ub.sort(reverse=True, key=lambda x: x[0])
            first_candidates = [pos for _, pos in candidates_with_ub]
        else:
            # Without pruning, we can still sort by single quality for better LB
            # This is a cheaper approximation
            candidates_with_quality = []
            for pos in first_candidates:
                if state.is_feasible(player, pos):
                    quality = state.board.single_quality.get(pos, 0.0)
                    candidates_with_quality.append((quality, pos))
            
            # Sort by quality descending (best first)
            candidates_with_quality.sort(reverse=True, key=lambda x: x[0])
            first_candidates = [pos for _, pos in candidates_with_quality]
        
        # Try each feasible first position (now sorted by quality/UB)
        for first_pos in first_candidates:
            # 1. Feasibility pruning (already done, but double-check)
            if not state.is_feasible(player, first_pos):
                if self.enable_pruning:
                    self.feasibility_prunings += 1
                continue
            
            # 2. Upper bound pruning (use cached value if available)
            if self.enable_pruning:
                UB = candidate_ubs.get(first_pos, 
                                      state.upper_bound_for_player_given_first(player, first_pos))
                if UB <= best_value:
                    # This branch cannot beat the best known value for this player
                    self.upper_bound_prunings += 1
                    continue
            
            # 3. Place first settlement
            s1 = state.clone()
            s1.place_settlement(player, first_pos)
            
            # 4. Recurse on later players
            s2 = self.dfs(player + 1, s1)
            if s2 is None:
                continue
            
            # 5. Now place the second settlement for this player (best complement)
            second_candidates = s2.get_feasible_positions(player)
            second_candidates = [v for v in second_candidates if v != first_pos]
            
            if not second_candidates:
                continue
            
            # Pick best second position using precomputed pair_quality
            best_second_pos = None
            best_two_house_value = -float('inf')
            
            for second_pos in second_candidates:
                val = s2.pair_quality(player, first_pos, second_pos)
                if val > best_two_house_value:
                    best_two_house_value = val
                    best_second_pos = second_pos
            
            if best_second_pos is None:
                continue
            
            s_final = s2.clone()
            s_final.place_settlement(player, best_second_pos)
            
            # 6. This branch payoff for this player (their own two-settlement benefit)
            branch_value = best_two_house_value
            
            # Update local LB
            if branch_value > best_value:
                best_value = branch_value
                best_state_for_player = s_final
        
        # Store in memo
        self.memo[memo_key] = best_value
        
        return best_state_for_player
    
    def solve(self) -> Tuple[Optional[State], Optional[Tuple[int, int]], Optional[float]]:
        """
        Solve for optimal settlement placements.
        
        Returns:
            Tuple of (final_state, player1_positions, player1_quality)
            where player1_positions is (first_pos, second_pos)
        """
        # Reset metrics
        self.recursive_calls = 0
        self.feasibility_prunings = 0
        self.upper_bound_prunings = 0
        self.memo_hits = 0
        self.memo_misses = 0
        self.memo.clear()
        
        # Start timer
        self.start_time = time.time()
        
        initial_state = State(self.board)
        final_state = self.dfs(player=1, state=initial_state)
        
        # End timer
        self.end_time = time.time()
        
        if final_state is None:
            return None, None, None
        
        # Get Player 1's positions and quality
        if len(final_state.houses[1]) != 2:
            return final_state, None, None
        
        first_pos, second_pos = final_state.houses[1]
        p1_quality = final_state.quality_of_player(1)
        
        return final_state, (first_pos, second_pos), p1_quality
    
    def get_metrics(self) -> dict:
        """
        Get performance metrics.
        
        Returns:
            Dictionary with metrics
        """
        elapsed_time = (self.end_time - self.start_time) if self.end_time and self.start_time else 0.0
        
        return {
            'recursive_calls': self.recursive_calls,
            'feasibility_prunings': self.feasibility_prunings,
            'upper_bound_prunings': self.upper_bound_prunings,
            'total_prunings': self.feasibility_prunings + self.upper_bound_prunings,
            'memo_hits': self.memo_hits,
            'memo_misses': self.memo_misses,
            'memo_size': len(self.memo),
            'memo_hit_rate': self.memo_hits / (self.memo_hits + self.memo_misses) if (self.memo_hits + self.memo_misses) > 0 else 0.0,
            'elapsed_time_seconds': elapsed_time,
            'pruning_enabled': self.enable_pruning
        }
    
    def print_metrics(self):
        """Print performance metrics in a readable format."""
        metrics = self.get_metrics()
        
        print("=" * 60)
        print("PERFORMANCE METRICS")
        print("=" * 60)
        print(f"Pruning enabled: {metrics['pruning_enabled']}")
        print(f"Elapsed time: {metrics['elapsed_time_seconds']:.4f} seconds")
        print()
        print(f"Recursive calls: {metrics['recursive_calls']:,}")
        print()
        print("Pruning statistics:")
        print(f"  Feasibility prunings: {metrics['feasibility_prunings']:,}")
        print(f"  Upper bound prunings: {metrics['upper_bound_prunings']:,}")
        print(f"  Total prunings: {metrics['total_prunings']:,}")
        if metrics['recursive_calls'] > 0:
            pruning_rate = metrics['total_prunings'] / metrics['recursive_calls'] * 100
            print(f"  Pruning rate: {pruning_rate:.2f}%")
        print()
        print("Memoization statistics:")
        print(f"  Memo hits: {metrics['memo_hits']:,}")
        print(f"  Memo misses: {metrics['memo_misses']:,}")
        print(f"  Memo size: {metrics['memo_size']:,}")
        if metrics['memo_hits'] + metrics['memo_misses'] > 0:
            print(f"  Hit rate: {metrics['memo_hit_rate']:.2%}")
        print()

