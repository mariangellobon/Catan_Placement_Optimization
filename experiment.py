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
    Uses the parent Solver's dfs method with configurable pruning options.
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
        # Call parent constructor with configurable pruning options
        super().__init__(board, enable_pruning=True,
                        enable_feasibility=enable_feasibility,
                        enable_upper_bound=enable_upper_bound,
                        enable_memo=enable_memo)
    
    # solve() method is inherited from Solver, no need to override
    # It already tracks metrics and timing
    
    def get_elapsed_time(self):
        """Get elapsed time in seconds."""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return 0.0


def run_experiment(num_boards=10, time_limit=25.0, modalities_to_test=None,
                  num_players=4, quality_weights=None):
    """
    Run experiment comparing different pruning modalities.
    
    Args:
        num_boards: Number of different boards to test
        time_limit: Maximum time per execution in seconds
        modalities_to_test: List of modality indices to test (0=feasibility only, 1=feasibility+memo, 2=all)
                          If None, tests all 3 modalities
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
            'name': 'Feasibility Pruning Only',
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
    
    # Filter modalities if specified
    if modalities_to_test is not None:
        selected_modalities = [modalities[i] for i in modalities_to_test if 0 <= i < len(modalities)]
        if not selected_modalities:
            print(f"Warning: No valid modalities selected. Using all modalities.")
            selected_modalities = modalities
        modalities = selected_modalities
    
    # FIRST: Generate all boards
    print("Generating boards...")
    print(f"  Configuration: {num_players} players")
    if quality_weights:
        print(f"  Quality weights: resources={quality_weights['w_resources']:.3f}, "
              f"expected_cards={quality_weights['w_expected_cards']:.3f}, "
              f"prob_at_least_one={quality_weights['w_prob_at_least_one']:.3f}")
    boards = []
    for board_num in range(num_boards):
        seed = board_num  # Use board number as seed for reproducibility
        board = Board(seed=seed, num_players=num_players, quality_weights=quality_weights)
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
        solutions = []  # Store solutions for comparison: (positions, quality)
        
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
                
                # Store solution for comparison
                solutions.append({
                    'board_num': board_num,
                    'positions': positions,
                    'quality': quality,
                    'final_state': final_state,
                    'recursive_calls': solver.recursive_calls
                })
                
                print(f"  Board {board_num}: {elapsed:.4f}s - "
                      f"Positions: {positions}, Objective value player one: {quality:.4f} - "
                      f"Recursive calls: {solver.recursive_calls:,}")
                
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
        
        # Calculate average recursive calls
        recursive_calls_list = [sol['recursive_calls'] for sol in solutions]
        if recursive_calls_list:
            avg_recursive_calls = sum(recursive_calls_list) / len(recursive_calls_list)
        else:
            avg_recursive_calls = 0
        
        results[modality['name']] = {
            'times': times,
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'timeouts': timeouts,
            'successful': successful,
            'total': num_boards,
            'solutions': solutions,  # Store solutions for comparison
            'avg_recursive_calls': avg_recursive_calls
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
    
    print(f"{'Modality':<50} {'Average (s)':<15} {'Min (s)':<12} {'Max (s)':<12} {'Timeouts':<10} {'Successful':<10} {'Recursive Calls':<20}")
    print("-" * 130)
    
    for modality_name, result in results.items():
        if result['times']:
            avg_str = f"{result['avg_time']:.4f}"
            min_str = f"{result['min_time']:.4f}"
            max_str = f"{result['max_time']:.4f}"
        else:
            avg_str = "N/A"
            min_str = "N/A"
            max_str = "N/A"
        
        recursive_calls_str = f"{result['avg_recursive_calls']:,.0f}" if result['avg_recursive_calls'] > 0 else "N/A"
        
        print(f"{modality_name:<50} {avg_str:<15} {min_str:<12} {max_str:<12} "
              f"{result['timeouts']:<10} {result['successful']:<10} {recursive_calls_str:<20}")
    
    print()
    
    # Compare solutions across modalities
    if len(results) >= 2:
        print("=" * 80)
        print("SOLUTION COMPARISON")
        print("=" * 80)
        print()
        
        # Get all modality names
        modality_names = list(results.keys())
        
        # Compare solutions board by board
        all_boards_same = True
        quality_differences = []
        
        # Find common boards (boards that were solved by all modalities)
        all_solutions = {}
        for mod_name, result in results.items():
            for sol in result['solutions']:
                board_num = sol['board_num']
                if board_num not in all_solutions:
                    all_solutions[board_num] = {}
                all_solutions[board_num][mod_name] = sol
        
        # Compare each board
        for board_num in sorted(all_solutions.keys()):
            board_solutions = all_solutions[board_num]
            
            # Check if all modalities solved this board
            if len(board_solutions) < len(modality_names):
                continue  # Skip if not all modalities solved it
            
            # Get first modality as reference
            ref_mod = modality_names[0]
            ref_sol = board_solutions[ref_mod]
            ref_positions = ref_sol['positions']
            ref_quality = ref_sol['quality']
            
            # Compare with other modalities
            board_matches = True
            for mod_name in modality_names[1:]:
                if mod_name not in board_solutions:
                    continue
                other_sol = board_solutions[mod_name]
                other_positions = other_sol['positions']
                other_quality = other_sol['quality']
                
                # Compare positions (should be same)
                if ref_positions != other_positions:
                    print(f"  Board {board_num}: POSITION MISMATCH!")
                    print(f"    {ref_mod}: {ref_positions}")
                    print(f"    {mod_name}: {other_positions}")
                    board_matches = False
                    all_boards_same = False
                
                # Compare objective value (should be same, but allow small floating point differences)
                quality_diff = abs(ref_quality - other_quality)
                if quality_diff > 1e-6:
                    print(f"  Board {board_num}: OBJECTIVE VALUE MISMATCH!")
                    print(f"    {ref_mod}: {ref_quality:.6f}")
                    print(f"    {mod_name}: {other_quality:.6f}")
                    print(f"    Difference: {quality_diff:.6f}")
                    board_matches = False
                    all_boards_same = False
                    quality_differences.append({
                        'board': board_num,
                        'mod1': ref_mod,
                        'mod2': mod_name,
                        'diff': quality_diff
                    })
            
            if board_matches:
                print(f"  Board {board_num}: ✓ All modalities agree - Positions: {ref_positions}, Objective value player one: {ref_quality:.6f}")
        
        print()
        if all_boards_same:
            print("✓ All solutions match across all modalities!")
        else:
            print("⚠ Some solutions differ between modalities!")
            if quality_differences:
                print(f"  Found {len(quality_differences)} objective value difference(s)")
        print()
        
        # Calculate speedup
        baseline = results[modalities[0]['name']]
        if baseline['times']:
            baseline_avg = baseline['avg_time']
            
            print("Speedup relative to first modality:")
            for modality_name, result in results.items():
                if modality_name != modalities[0]['name'] and result['times']:
                    speedup = baseline_avg / result['avg_time']
                    print(f"  {modality_name}: {speedup:.2f}x faster")
    
    return results


if __name__ == "__main__":
    # Parse command line arguments
    num_boards = 10
    time_limit = 25.0
    modalities_to_test = None
    
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
    
    # Parse modality selection (--modalities 0,1,2 or --modalities 0,2)
    if len(sys.argv) > 3:
        if sys.argv[3].startswith('--modalities='):
            mod_str = sys.argv[3].split('=', 1)[1]
        elif sys.argv[3] == '--modalities' and len(sys.argv) > 4:
            mod_str = sys.argv[4]
        else:
            mod_str = None
        
        if mod_str:
            try:
                modalities_to_test = [int(x.strip()) for x in mod_str.split(',')]
                print(f"Selected modalities: {modalities_to_test}")
                print("  0 = Feasibility Pruning Only")
                print("  1 = Feasibility + Memo")
                print("  2 = All Prunings")
                print()
            except ValueError:
                print(f"Invalid modality selection: {mod_str}. Using all modalities.")
                modalities_to_test = None
    
    # Parse additional arguments (players and weights)
    num_players = 4
    quality_weights = None
    
    for i in range(3, len(sys.argv)):
        arg = sys.argv[i]
        if arg.startswith('--players=') or arg.startswith('-p='):
            try:
                num_players = int(arg.split('=', 1)[1])
                if num_players < 2 or num_players > 4:
                    print(f"Error: Number of players must be between 2 and 4. Got {num_players}")
                    sys.exit(1)
            except ValueError:
                print(f"Error: Invalid number of players: {arg.split('=', 1)[1]}")
                sys.exit(1)
        elif arg.startswith('--weights=') or arg.startswith('-w='):
            try:
                weights_str = arg.split('=', 1)[1]
                weights_list = [float(w.strip()) for w in weights_str.split(',')]
                if len(weights_list) != 3:
                    print(f"Error: Must provide exactly 3 weights. Got {len(weights_list)}")
                    sys.exit(1)
                quality_weights = {
                    'w_resources': weights_list[0],
                    'w_expected_cards': weights_list[1],
                    'w_prob_at_least_one': weights_list[2]
                }
                # Normalize weights
                total = sum(weights_list)
                if total > 0:
                    quality_weights['w_resources'] /= total
                    quality_weights['w_expected_cards'] /= total
                    quality_weights['w_prob_at_least_one'] /= total
            except ValueError as e:
                print(f"Error: Invalid weights format. Expected --weights=w1,w2,w3. Error: {e}")
                sys.exit(1)
    
    # Run experiment
    results = run_experiment(num_boards=num_boards, time_limit=time_limit, 
                            modalities_to_test=modalities_to_test,
                            num_players=num_players, quality_weights=quality_weights)

