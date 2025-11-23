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
    
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg in ["--no-pruning", "-n", "--compare", "-c"]:
                compare_no_pruning = True
            elif arg.startswith("--save="):
                save_image = arg.split("=", 1)[1]
            elif arg.startswith("-s="):
                save_image = arg.split("=", 1)[1]
            elif arg in ["--help", "-h"]:
                print("Usage: python main.py [seed] [--compare|-c] [--save=path.png]")
                print("  seed: Optional random seed for reproducibility")
                print("  --compare, -c: Also run without pruning for comparison")
                print("  --save=path.png, -s=path.png: Save visualization to file")
                return
            else:
                try:
                    seed = int(arg)
                except ValueError:
                    print(f"Unknown argument: {arg}. Use --help for usage.")
                    return
    
    print("Initializing Catan board...")
    board = Board(seed=seed)
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
    print(f"Player 1 quality score: {player1_quality:.4f}")
    print()
    
    # Display all players' placements
    print("All players' settlements:")
    for player in range(1, 5):
        houses = final_state.houses[player]
        if len(houses) == 2:
            quality = final_state.quality_of_player(player)
            print(f"  Player {player}: vertices {houses}, quality: {quality:.4f}")
        else:
            print(f"  Player {player}: vertices {houses} (incomplete)")
    print()
    
    # Display metrics
    solver.print_metrics()
    
    # Visualize the board (console)
    print("\n")
    visualize_board(board, final_state)
    print("\n")
    visualize_settlements_simple(board, final_state)
    
    # Visualize the board (GUI)
    print("\nGenerating graphical visualization...")
    try:
        visualize_board_gui(board, final_state, save_path=save_image)
        if save_image:
            print(f"Graphical visualization saved to {save_image}")
        else:
            print("Graphical visualization displayed.")
    except ImportError:
        print("Warning: matplotlib not available. Install with: pip install matplotlib")
    except Exception as e:
        print(f"Error generating GUI visualization: {e}")
        import traceback
        traceback.print_exc()
        print("Falling back to console visualization only.")
    
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

