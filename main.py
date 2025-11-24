"""
Main entry point for Catan optimal settlement placement solver.

Initializes a random board, runs the solver, and displays results.
"""

import sys
from board import Board
from solver import Solver
from visualization import visualize_board, visualize_settlements_simple
from visualization_gui import visualize_board_gui, visualize_settlements_detailed


def main():
    """Main function to run the solver."""
    # Parse command line arguments
    seed = None
    compare_no_pruning = False
    save_image = None
    num_players = 4
    quality_weights = None
    
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg in ["--no-pruning", "-n", "--compare", "-c"]:
                compare_no_pruning = True
            elif arg.startswith("--save="):
                save_image = arg.split("=", 1)[1]
            elif arg.startswith("-s="):
                save_image = arg.split("=", 1)[1]
            elif arg.startswith("--players=") or arg.startswith("-p="):
                try:
                    num_players = int(arg.split("=", 1)[1])
                    if num_players < 2 or num_players > 4:
                        print(f"Error: Number of players must be between 2 and 4. Got {num_players}")
                        return
                except ValueError:
                    print(f"Error: Invalid number of players: {arg.split('=', 1)[1]}")
                    return
            elif arg.startswith("--weights=") or arg.startswith("-w="):
                try:
                    weights_str = arg.split("=", 1)[1]
                    weights_list = [float(w.strip()) for w in weights_str.split(",")]
                    if len(weights_list) != 3:
                        print(f"Error: Must provide exactly 3 weights (w_resources, w_expected_cards, w_prob_at_least_one). Got {len(weights_list)}")
                        return
                    quality_weights = {
                        'w_resources': weights_list[0],
                        'w_expected_cards': weights_list[1],
                        'w_prob_at_least_one': weights_list[2]
                    }
                    # Normalize weights (optional, but good practice)
                    total = sum(weights_list)
                    if total > 0:
                        quality_weights['w_resources'] /= total
                        quality_weights['w_expected_cards'] /= total
                        quality_weights['w_prob_at_least_one'] /= total
                except ValueError as e:
                    print(f"Error: Invalid weights format. Expected --weights=w1,w2,w3. Error: {e}")
                    return
            elif arg in ["--help", "-h"]:
                print("Usage: python main.py [seed] [options]")
                print("  seed: Optional random seed for reproducibility")
                print("  --compare, -c: Also run without pruning for comparison")
                print("  --save=path.png, -s=path.png: Save visualization to file")
                print("  --players=N, -p=N: Number of players (2-4, default: 4)")
                print("  --weights=w1,w2,w3, -w=w1,w2,w3: Quality function weights")
                print("     w1 = weight for resource score")
                print("     w2 = weight for expected cards")
                print("     w3 = weight for probability at least one")
                print("     (weights will be normalized to sum to 1)")
                return
            else:
                try:
                    seed = int(arg)
                except ValueError:
                    print(f"Unknown argument: {arg}. Use --help for usage.")
                    return
    
    print("Initializing Catan board...")
    board = Board(seed=seed, num_players=num_players, quality_weights=quality_weights)
    print(f"Board created with {len(board.tiles)} tiles and {len(board.vertices)} vertices")
    print(f"Resources: {[t['resource'] for t in board.tiles]}")
    print()
    
    # Run with pruning
    print("Creating solver (with pruning)...")
    solver = Solver(board, enable_pruning=True)
    print()
    
    print("Solving for optimal settlement placements (WITH PRUNING)...")
    print("This may take a while depending on board size and pruning effectiveness...")
    print()
    
    final_state, player1_positions, player1_quality = solver.solve()
    
    if final_state is None:
        print("ERROR: No valid solution found!")
        return
    
    if player1_positions is None:
        print("ERROR: Player 1 does not have 2 settlements!")
        return
    
    print("=" * 60)
    print("SOLUTION FOUND (WITH PRUNING)")
    print("=" * 60)
    print()
    
    print(f"Player 1 optimal positions: {player1_positions}")
    print(f"Player 1  objetive value: {player1_quality:.4f}")
    print()
    
    # Display all players' placements
    print("All players' settlements:")
    for player in range(1, num_players + 1):
        houses = final_state.houses[player]
        if len(houses) == 2:
            quality = final_state.quality_of_player(player)
            print(f"  Player {player}: vertices {houses}, objective value: {quality:.4f}")
        else:
            print(f"  Player {player}: vertices {houses} (incomplete)")
    print()
    
    # Display metrics
    solver.print_metrics()
    
    # Visualize the board (GUI)
    print("\nGenerating graphical visualization...")
    try:
        visualize_board_gui(board, final_state, save_path=save_image)
    except ImportError:
        print("Warning: matplotlib not available. Install with: pip install matplotlib")
    except Exception as e:
        print(f"Error generating GUI visualization: {e}")
        import traceback
        traceback.print_exc()
    
    # Run without pruning for comparison
    if compare_no_pruning:
        print("\n" + "=" * 60)
        print("RUNNING WITHOUT PRUNING FOR COMPARISON")
        print("=" * 60)
        print("This will take significantly longer...")
        print()
        
        solver_no_pruning = Solver(board, enable_pruning=False)
        final_state_np, player1_positions_np, player1_quality_np = solver_no_pruning.solve()
        
        if final_state_np is not None and player1_positions_np is not None:
            print("\n" + "=" * 60)
            print("SOLUTION FOUND (WITHOUT PRUNING)")
            print("=" * 60)
            print()
            print(f"Player 1 optimal positions: {player1_positions_np}")
            print(f"Player 1 quality score: {player1_quality_np:.4f}")
            print()
            
            # Verify solutions match
            if player1_positions == player1_positions_np:
                print("✓ Solutions match!")
            else:
                print("⚠ Solutions differ!")
            print()
            
            # Display metrics
            solver_no_pruning.print_metrics()
            
            # Comparison
            print("\n" + "=" * 60)
            print("COMPARISON")
            print("=" * 60)
            metrics = solver.get_metrics()
            metrics_np = solver_no_pruning.get_metrics()
            
            print(f"Time with pruning:    {metrics['elapsed_time_seconds']:.4f} seconds")
            print(f"Time without pruning: {metrics_np['elapsed_time_seconds']:.4f} seconds")
            if metrics_np['elapsed_time_seconds'] > 0:
                speedup = metrics_np['elapsed_time_seconds'] / metrics['elapsed_time_seconds']
                print(f"Speedup: {speedup:.2f}x faster with pruning")
            print()
            
            print(f"Recursive calls with pruning:    {metrics['recursive_calls']:,}")
            print(f"Recursive calls without pruning: {metrics_np['recursive_calls']:,}")
            if metrics_np['recursive_calls'] > 0:
                reduction = (1 - metrics['recursive_calls'] / metrics_np['recursive_calls']) * 100
                print(f"Reduction: {reduction:.2f}%")
            print()
            
            print(f"Total prunings: {metrics['total_prunings']:,}")
            print()


if __name__ == "__main__":
    main()

