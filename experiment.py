"""
Experiment to compare different pruning modalities in the Catan solver.

Compares:
1. Feasibility pruning only
2. Feasibility pruning + memo
3. All prunings (feasibility + upper bound + memo)
"""

import time
import sys
import threading
from board import Board
from solver import Solver


class ExperimentSolver(Solver):
    """
    Extension of Solver to allow different pruning modalities.
    """
    
    def __init__(self, board: Board, enable_feasibility: bool = True,
                 enable_upper_bound: bool = True, enable_memo: bool = True):
        """
        Initialize solver with configurable pruning options.
        
        Args:
            board: Board object
            enable_feasibility: Enable feasibility pruning
            enable_upper_bound: Enable upper bound pruning
            enable_memo: Enable memoization
        """
        self.board = board
        self.enable_feasibility = enable_feasibility
        self.enable_upper_bound = enable_upper_bound
        self.enable_memo = enable_memo
        
        # Metrics
        self.recursive_calls = 0
        self.feasibility_prunings = 0
        self.upper_bound_prunings = 0
        self.memo_hits = 0
        self.memo_misses = 0
        self.start_time = None
        self.end_time = None
        
        # Memo (only if enabled)
        self.memo = {} if enable_memo else None
    
    def dfs(self, player: int, state):
        """DFS with configurable pruning."""
        self.recursive_calls += 1
        
        # Base case
        if player > 4:
            return state
        
        # Memoization (if enabled)
        if self.enable_memo:
            state_key = state.make_key()
            memo_key = (player, state_key)
            
            if memo_key in self.memo:
                self.memo_hits += 1
                # Return memoized value (we still need to explore, but can use for pruning)
            else:
                self.memo_misses += 1
        
        # Local lower bound
        best_value = -float('inf')
        best_state_for_player = None
        
        # Get feasible positions
        first_candidates = state.get_feasible_positions(player)
        
        if not first_candidates:
            if self.enable_memo:
                memo_key = (player, state.make_key())
                self.memo[memo_key] = -float('inf')
            return None
        
        # Sort candidates by upper bound or quality (if upper bound enabled)
        candidate_ubs = {}
        
        if self.enable_upper_bound:
            # Sort by upper bound
            candidates_with_ub = []
            for pos in first_candidates:
                if state.is_feasible(player, pos):
                    ub = state.upper_bound_for_player_given_first(player, pos)
                    candidate_ubs[pos] = ub
                    candidates_with_ub.append((ub, pos))
            
            candidates_with_ub.sort(reverse=True, key=lambda x: x[0])
            first_candidates = [pos for _, pos in candidates_with_ub]
        else:
            # Sort by single quality
            candidates_with_quality = []
            for pos in first_candidates:
                if state.is_feasible(player, pos):
                    quality = state.board.single_quality.get(pos, 0.0)
                    candidates_with_quality.append((quality, pos))
            
            candidates_with_quality.sort(reverse=True, key=lambda x: x[0])
            first_candidates = [pos for _, pos in candidates_with_quality]
        
        # Try each candidate
        for first_pos in first_candidates:
            # Feasibility pruning
            if not state.is_feasible(player, first_pos):
                if self.enable_feasibility:
                    self.feasibility_prunings += 1
                continue
            
            # Upper bound pruning
            if self.enable_upper_bound:
                UB = candidate_ubs.get(first_pos,
                                      state.upper_bound_for_player_given_first(player, first_pos))
                if UB <= best_value:
                    self.upper_bound_prunings += 1
                    continue
            
            # Place first settlement
            s1 = state.clone()
            s1.place_settlement(player, first_pos)
            
            # Recurse
            s2 = self.dfs(player + 1, s1)
            if s2 is None:
                continue
            
            # Place second settlement
            second_candidates = s2.get_feasible_positions(player)
            second_candidates = [v for v in second_candidates if v != first_pos]
            
            if not second_candidates:
                continue
            
            # Pick best second position
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
            
            branch_value = best_two_house_value
            
            if branch_value > best_value:
                best_value = branch_value
                best_state_for_player = s_final
        
        # Store in memo (if enabled)
        if self.enable_memo:
            memo_key = (player, state.make_key())
            self.memo[memo_key] = best_value
        
        return best_state_for_player
    
    def solve(self):
        """Solve with metrics tracking."""
        # Reset metrics
        self.recursive_calls = 0
        self.feasibility_prunings = 0
        self.upper_bound_prunings = 0
        self.memo_hits = 0
        self.memo_misses = 0
        if self.enable_memo:
            self.memo.clear()
        
        # Start timer
        self.start_time = time.time()
        
        from state import State
        initial_state = State(self.board)
        final_state = self.dfs(player=1, state=initial_state)
        
        # End timer
        self.end_time = time.time()
        
        if final_state is None:
            return None, None, None
        
        if len(final_state.houses[1]) != 2:
            return final_state, None, None
        
        first_pos, second_pos = final_state.houses[1]
        p1_quality = final_state.quality_of_player(1)
        
        return final_state, (first_pos, second_pos), p1_quality
    
    def get_elapsed_time(self):
        """Get elapsed time in seconds."""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return 0.0


def run_experiment(num_boards=10, time_limit=25.0):
    """
    Run experiment comparing different pruning modalities.
    
    Args:
        num_boards: Number of different boards to test
        time_limit: Maximum time per execution in seconds
    """
    print("=" * 80)
    print("EXPERIMENT: Comparison of Pruning Modalities")
    print("=" * 80)
    print(f"Generating {num_boards} boards and evaluating each with the 3 modalities")
    print(f"Time limit per execution: {time_limit} seconds")
    print()
    
    # Modalities to test
    modalities = [
        {
            'name': 'Solo Feasibility Pruning',
            'feasibility': True,
            'upper_bound': False,
            'memo': False
        },
        {
            'name': 'Feasibility + Memo',
            'feasibility': True,
            'upper_bound': False,
            'memo': True
        },
        {
            'name': 'All Prunings (Feasibility + Upper Bound + Memo)',
            'feasibility': True,
            'upper_bound': True,
            'memo': True
        }
    ]
    
    # FIRST: Generate all boards
    print("Generating boards...")
    boards = []
    for board_num in range(num_boards):
        seed = board_num  # Use board number as seed for reproducibility
        board = Board(seed=seed)
        boards.append(board)
        print(f"  Board {board_num} generated (seed={seed})")
    print()
    
    results = {}
    
    # SECOND: Evaluate each modality on the same boards
    for modality in modalities:
        print(f"\n{'='*80}")
        print(f"Modality: {modality['name']}")
        print(f"{'='*80}")
        
        times = []
        timeouts = 0
        successful = 0
        
        for board_num, board in enumerate(boards):
            try:
                # Create solver with this modality
                solver = ExperimentSolver(
                    board,
                    enable_feasibility=modality['feasibility'],
                    enable_upper_bound=modality['upper_bound'],
                    enable_memo=modality['memo']
                )
                
                # Run with timeout using threading
                result_container = {'result': None, 'exception': None, 'completed': False}
                
                def solve_wrapper():
                    try:
                        result_container['result'] = solver.solve()
                        result_container['completed'] = True
                    except Exception as e:
                        result_container['exception'] = e
                        result_container['completed'] = True
                
                # Start solver in a thread
                start = time.time()
                thread = threading.Thread(target=solve_wrapper)
                thread.daemon = True
                thread.start()
                thread.join(timeout=time_limit)
                
                elapsed = time.time() - start
                
                # Check if timeout occurred
                if thread.is_alive():
                    print(f"  Board {board_num}: TIMEOUT ({elapsed:.2f}s > {time_limit}s) - Execution interrupted")
                    timeouts += 1
                    continue
                
                # Check for exceptions
                if result_container['exception']:
                    print(f"  Board {board_num}: ERROR - {result_container['exception']}")
                    continue
                
                # Get result
                final_state, positions, quality = result_container['result']
                
                if final_state is None or positions is None:
                    print(f"  Board {board_num}: ERROR - No solution found")
                    continue
                
                times.append(elapsed)
                successful += 1
                print(f"  Board {board_num}: {elapsed:.4f}s - "
                      f"Recursive calls: {solver.recursive_calls:,}, "
                      f"Prunings: {solver.feasibility_prunings + solver.upper_bound_prunings:,}")
                
            except Exception as e:
                print(f"  Board {board_num}: ERROR - {e}")
                continue
        
        # Calculate statistics
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
        else:
            avg_time = float('inf')
            min_time = float('inf')
            max_time = float('inf')
        
        results[modality['name']] = {
            'times': times,
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'timeouts': timeouts,
            'successful': successful,
            'total': num_boards
        }
        
        print(f"\n  Summary:")
        print(f"    Successful: {successful}/{num_boards}")
        print(f"    Timeouts: {timeouts}/{num_boards}")
        if times:
            print(f"    Average time: {avg_time:.4f}s")
            print(f"    Min time: {min_time:.4f}s")
            print(f"    Max time: {max_time:.4f}s")
        else:
            print(f"    Average time: N/A (no successful executions)")
    
    # Print final comparison
    print("\n" + "=" * 80)
    print("COMPARATIVE SUMMARY")
    print("=" * 80)
    print()
    
    print(f"{'Modality':<50} {'Average (s)':<15} {'Min (s)':<12} {'Max (s)':<12} {'Timeouts':<10} {'Successful':<10}")
    print("-" * 110)
    
    for modality_name, result in results.items():
        if result['times']:
            avg_str = f"{result['avg_time']:.4f}"
            min_str = f"{result['min_time']:.4f}"
            max_str = f"{result['max_time']:.4f}"
        else:
            avg_str = "N/A"
            min_str = "N/A"
            max_str = "N/A"
        
        print(f"{modality_name:<50} {avg_str:<15} {min_str:<12} {max_str:<12} "
              f"{result['timeouts']:<10} {result['successful']:<10}")
    
    print()
    
    # Calculate speedup
    if len(results) >= 2:
        baseline = results[modalities[0]['name']]
        if baseline['times']:
            baseline_avg = baseline['avg_time']
            
            print("Speedup relative to 'Feasibility Pruning Only':")
            for modality_name, result in results.items():
                if modality_name != modalities[0]['name'] and result['times']:
                    speedup = baseline_avg / result['avg_time']
                    print(f"  {modality_name}: {speedup:.2f}x faster")
    
    return results


if __name__ == "__main__":
    # Parse command line arguments
    num_boards = 10
    time_limit = 25.0
    
    if len(sys.argv) > 1:
        try:
            num_boards = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number of boards: {sys.argv[1]}. Using default: 10")
    
    if len(sys.argv) > 2:
        try:
            time_limit = float(sys.argv[2])
        except ValueError:
            print(f"Invalid time limit: {sys.argv[2]}. Using default: 25.0")
    
    # Run experiment
    results = run_experiment(num_boards=num_boards, time_limit=time_limit)

