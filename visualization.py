"""
Visualization of Catan board with player settlements.
"""

from typing import Dict, List, Optional
from state import State
from board import Board


def visualize_board(board: Board, state: Optional[State] = None):
    """
    Visualize the Catan board with tiles, resources, numbers, and player settlements.
    
    Args:
        board: Board object
        state: Optional State object to show player settlements
    """
    print("=" * 80)
    print("CATAN BOARD VISUALIZATION")
    print("=" * 80)
    print()
    
    # Create a mapping of vertex to player for quick lookup
    vertex_to_player = {}
    if state:
        for player, vertices in state.houses.items():
            for vertex in vertices:
                vertex_to_player[vertex] = player
    
    # Resource abbreviations and colors (using ASCII symbols for Windows compatibility)
    resource_symbols = {
        'wood': 'W',
        'brick': 'B',
        'wheat': 'G',  # Grain
        'ore': 'O',
        'sheep': 'S',
        'desert': 'D'
    }
    
    # Player colors/symbols (using ASCII)
    player_symbols = {
        1: 'R',  # Red
        2: 'U',  # Blue
        3: 'G',  # Green
        4: 'Y',  # Yellow
    }
    
    # Print board layout
    print("Board Layout (Standard Catan - 19 tiles):")
    print()
    
    # Group tiles by row
    tiles_by_row = {}
    for tile in board.tiles:
        row = tile['row']
        if row not in tiles_by_row:
            tiles_by_row[row] = []
        tiles_by_row[row].append(tile)
    
    # Sort tiles in each row by column
    for row in tiles_by_row:
        tiles_by_row[row].sort(key=lambda t: t['col'])
    
    # Print tiles in hexagonal layout
    for row in sorted(tiles_by_row.keys()):
        tiles = tiles_by_row[row]
        
        # Indent based on row (hexagonal pattern)
        indent = "  " * (5 - len(tiles))
        print(indent, end="")
        
        for tile in tiles:
            resource = tile['resource']
            number = tile['number']
            symbol = resource_symbols.get(resource, '?')
            
            # Format tile display
            if number is not None:
                tile_str = f"{symbol} {number:2d}"
            else:
                tile_str = f"{symbol} --"
            
            print(f" [{tile_str}]", end="")
        print()
    
    print()
    
    # Show tile details
    print("Tile Details:")
    print("-" * 80)
    for tile in sorted(board.tiles, key=lambda t: (t['row'], t['col'])):
        resource = tile['resource']
        number = tile['number']
        symbol = resource_symbols.get(resource, '?')
        
        if number is not None:
            print(f"  Tile {tile['id']:2d} (Row {tile['row']}, Col {tile['col']}): "
                  f"{symbol} {resource:8s} - Number: {number:2d}")
        else:
            print(f"  Tile {tile['id']:2d} (Row {tile['row']}, Col {tile['col']}): "
                  f"{symbol} {resource:8s} - Number: --")
    print()
    
    # Show player settlements if state is provided
    if state:
        print("=" * 80)
        print("PLAYER SETTLEMENTS")
        print("=" * 80)
        print()
        
        for player in sorted(state.houses.keys()):
            vertices = state.houses[player]
            if len(vertices) == 2:
                symbol = player_symbols.get(player, '.')
                quality = state.quality_of_player(player)
                
                print(f"{symbol} Player {player}:")
                print(f"   Settlement 1: Vertex {vertices[0]}")
                print(f"   Settlement 2: Vertex {vertices[1]}")
                print(f"   Quality Score: {quality:.4f}")
                
                # Show which tiles each settlement touches
                print(f"   Settlement 1 touches tiles: {board.tiles_touching.get(vertices[0], [])}")
                print(f"   Settlement 2 touches tiles: {board.tiles_touching.get(vertices[1], [])}")
                print()
        
        # Create vertex occupancy map
        print("Vertex Occupancy Map:")
        print("-" * 80)
        
        # Group vertices by which tiles they touch (for visualization)
        occupied_vertices = {}
        for player, vertices in state.houses.items():
            for vertex in vertices:
                occupied_vertices[vertex] = player
        
        # Show first 20 vertices as example
        print("First 30 vertices:")
        for vertex in sorted(board.vertices)[:30]:
            if vertex in occupied_vertices:
                player = occupied_vertices[vertex]
                symbol = player_symbols.get(player, '.')
                tiles = board.tiles_touching.get(vertex, [])
                print(f"  Vertex {vertex:2d}: [{symbol}] Player {player} "
                      f"(touches tiles: {tiles})")
            else:
                tiles = board.tiles_touching.get(vertex, [])
                print(f"  Vertex {vertex:2d}: [.] Available (touches tiles: {tiles})")
        
        if len(board.vertices) > 30:
            print(f"  ... and {len(board.vertices) - 30} more vertices")
        print()
    
    # Summary statistics
    print("=" * 80)
    print("BOARD STATISTICS")
    print("=" * 80)
    print(f"Total tiles: {len(board.tiles)}")
    print(f"Total vertices: {len(board.vertices)}")
    
    if state:
        occupied = sum(1 for v, p in state.occupied.items() if p is not None)
        print(f"Occupied vertices: {occupied}")
        print(f"Available vertices: {len(state.available_vertices)}")
    
    print()


def visualize_settlements_simple(board: Board, state: State):
    """
    Simple visualization focusing on player settlements.
    
    Args:
        board: Board object
        state: State object with player settlements
    """
    print("=" * 80)
    print("SETTLEMENT PLACEMENT SUMMARY")
    print("=" * 80)
    print()
    
    player_symbols = {
        1: 'R',  # Red
        2: 'U',  # Blue
        3: 'G',  # Green
        4: 'Y',  # Yellow
    }
    
    resource_symbols = {
        'wood': 'W',
        'brick': 'B',
        'wheat': 'G',  # Grain
        'ore': 'O',
        'sheep': 'S',
        'desert': 'D'
    }
    
    for player in sorted(state.houses.keys()):
        vertices = state.houses[player]
        if len(vertices) != 2:
            continue
        
        symbol = player_symbols.get(player, '.')
        quality = state.quality_of_player(player)
        
        print(f"{symbol} PLAYER {player}")
        print("-" * 80)
        
        for i, vertex in enumerate(vertices, 1):
            print(f"  Settlement {i}: Vertex {vertex}")
            
            # Get tiles this vertex touches
            tile_ids = board.tiles_touching.get(vertex, [])
            print(f"    Touches {len(tile_ids)} tiles:")
            
            for tile_id in tile_ids:
                if tile_id < len(board.tiles):
                    tile = board.tiles[tile_id]
                    resource = tile['resource']
                    number = tile['number']
                    res_symbol = resource_symbols.get(resource, '?')
                    
                    if number is not None:
                        print(f"      - Tile {tile_id:2d}: {res_symbol} {resource:8s} (Number: {number:2d})")
                    else:
                        print(f"      - Tile {tile_id:2d}: {res_symbol} {resource:8s} (Desert)")
        
        print(f"  Quality Score: {quality:.4f}")
        print()

