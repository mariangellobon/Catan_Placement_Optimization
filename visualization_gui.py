"""
Visual GUI visualization of Catan board with player settlements.
Uses matplotlib to create a graphical representation.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import RegularPolygon, Circle
import numpy as np
from typing import Optional, Dict
from state import State
from board import Board


def hex_to_pixel(row: int, col: int, size: float = 1.0) -> tuple:
    """
    Convert hexagonal grid coordinates to pixel coordinates for Catan layout.
    
    Catan has an irregular layout:
    - Row 0: 3 tiles (cols 0, 1, 2) - tiles 0, 1, 2
    - Row 1: 4 tiles (cols 0, 1, 2, 3) - tiles 3, 4, 5, 6
    - Row 2: 5 tiles (cols 0, 1, 2, 3, 4) - tiles 7, 8, 9, 10, 11
    - Row 3: 4 tiles (cols 0, 1, 2, 3) - tiles 12, 13, 14, 15
    - Row 4: 3 tiles (cols 0, 1, 2) - tiles 16, 17, 18
    
    Args:
        row: Row in hexagonal grid (0-4)
        col: Column in hexagonal grid
        size: Size of hexagon
        
    Returns:
        (x, y) pixel coordinates
    """
    # Row offsets to center the irregular layout
    row_offsets = {
        0: 1.0,   # Row 0 (3 tiles) - offset right
        1: 0.5,   # Row 1 (4 tiles) - slight offset
        2: 0.0,   # Row 2 (5 tiles) - centered
        3: 0.5,   # Row 3 (4 tiles) - slight offset
        4: 1.0    # Row 4 (3 tiles) - offset right
    }
    
    # Base offset for this row
    offset = row_offsets.get(row, 0.0)
    
    # Calculate position with offset
    x = size * (np.sqrt(3) * (col + offset))
    y = size * (3/2 * row)
    return (x, y)


def vertex_to_pixel(vertex_id: int, tile_centers: Dict[int, tuple], 
                    board: Board, hex_radius: float) -> tuple:
    """
    Calculate pixel position for a vertex based on tiles it touches.
    
    Args:
        vertex_id: Vertex ID (0-53)
        tile_centers: Dictionary mapping tile_id -> (x, y)
        board: Board object
        hex_radius: Radius of hexagons
        
    Returns:
        (x, y) pixel coordinates for the vertex
    """
    tile_ids = board.tiles_touching.get(vertex_id, [])
    if not tile_ids:
        return (0, 0)
    
    # Get positions of tiles this vertex touches
    positions = [tile_centers[tid] for tid in tile_ids if tid in tile_centers]
    if not positions:
        return (0, 0)
    
    if len(positions) == 1:
        # Vertex touches one tile - position on edge
        tile_x, tile_y = positions[0]
        # Place on top edge of hexagon
        return (tile_x, tile_y + hex_radius)
    elif len(positions) == 2:
        # Vertex touches two tiles - position between them
        p1, p2 = positions[0], positions[1]
        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2
        # Move outward from center along the line between tiles
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = np.sqrt(dx*dx + dy*dy)
        if dist > 0:
            # Normalize and move outward to edge
            return (mid_x + hex_radius*0.6*dx/dist, mid_y + hex_radius*0.6*dy/dist)
        else:
            return (mid_x, mid_y)
    else:
        # Vertex touches 3 tiles - use average (corner vertex)
        avg_x = sum(p[0] for p in positions) / len(positions)
        avg_y = sum(p[1] for p in positions) / len(positions)
        return (avg_x, avg_y)


def visualize_board_gui(board: Board, state: Optional[State] = None, 
                        save_path: Optional[str] = None):
    """
    Create a graphical visualization of the Catan board.
    
    Args:
        board: Board object
        state: Optional State object to show player settlements
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Resource colors
    resource_colors = {
        'wood': '#8B4513',      # Brown
        'brick': '#CD5C5C',      # Reddish
        'wheat': '#FFD700',      # Gold
        'ore': '#708090',        # Gray
        'sheep': '#90EE90',      # Light green
        'desert': '#F5DEB3'      # Beige
    }
    
    # Player colors
    player_colors = {
        1: '#FF0000',  # Red
        2: '#0000FF',  # Blue
        3: '#00FF00',  # Green
        4: '#FFFF00',  # Yellow
    }
    
    # Resource abbreviations
    resource_abbrev = {
        'wood': 'W',
        'brick': 'B',
        'wheat': 'G',
        'ore': 'O',
        'sheep': 'S',
        'desert': 'D'
    }
    
    # Draw tiles
    tile_centers = {}
    hex_radius = 0.8  # Smaller radius to avoid overlap
    
    for tile in board.tiles:
        row, col = tile['row'], tile['col']
        x, y = hex_to_pixel(row, col, size=hex_radius)
        tile_centers[tile['id']] = (x, y)
        
        resource = tile['resource']
        color = resource_colors.get(resource, '#CCCCCC')
        
        # Draw hexagon with proper orientation (rotated 30 degrees)
        hexagon = RegularPolygon((x, y), numVertices=6, radius=hex_radius,
                                orientation=np.pi/6 + np.pi/6, facecolor=color,
                                edgecolor='black', linewidth=2, alpha=0.9)
        ax.add_patch(hexagon)
        
        # Add resource label
        abbrev = resource_abbrev.get(resource, '?')
        label_y = y + hex_radius * 0.4
        ax.text(x, label_y, abbrev, ha='center', va='center',
               fontsize=11, fontweight='bold', 
               color='white' if resource != 'desert' else 'black',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.3) if resource == 'desert' else None)
        
        # Add number token
        number = tile['number']
        if number is not None:
            # Draw circle for number
            circle_radius = hex_radius * 0.25
            circle = Circle((x, y - hex_radius * 0.4), circle_radius, 
                           facecolor='white', edgecolor='black', linewidth=1.5)
            ax.add_patch(circle)
            ax.text(x, y - hex_radius * 0.4, str(number), ha='center', va='center',
                   fontsize=9, fontweight='bold')
    
    # Draw vertices and settlements
    if state:
        # Create vertex positions using exact tile mappings
        vertex_positions = {}
        
        # For each vertex, calculate position based on tiles it touches
        for vertex in board.vertices:
            vx, vy = vertex_to_pixel(vertex, tile_centers, board, hex_radius)
            vertex_positions[vertex] = (vx, vy)
        
        # Draw all vertices as small dots
        for vertex, (vx, vy) in vertex_positions.items():
            # Check if this vertex has a settlement
            has_settlement = False
            settlement_player = None
            
            if state:
                for player, vertices in state.houses.items():
                    if vertex in vertices:
                        has_settlement = True
                        settlement_player = player
                        break
            
            if has_settlement:
                # Draw settlement (larger circle with player color)
                color = player_colors.get(settlement_player, '#000000')
                settlement_radius = hex_radius * 0.15
                settlement = Circle((vx, vy), settlement_radius, facecolor=color,
                                   edgecolor='black', linewidth=2.5, zorder=10)
                ax.add_patch(settlement)
                # Add player number
                ax.text(vx, vy, str(settlement_player), ha='center', va='center',
                       fontsize=9, fontweight='bold', color='white', zorder=11)
            else:
                # Draw empty vertex (small dot) - optional, can comment out for cleaner look
                # vertex_dot = Circle((vx, vy), hex_radius * 0.05, facecolor='gray',
                #                   edgecolor='black', linewidth=1, alpha=0.3)
                # ax.add_patch(vertex_dot)
                pass
    
    # Set title
    title = "Catan Board - Optimal Settlement Placement"
    if state:
        title += "\n"
        player_info = []
        for player in sorted(state.houses.keys()):
            vertices = state.houses[player]
            if len(vertices) == 2:
                quality = state.quality_of_player(player)
                color_name = ['Red', 'Blue', 'Green', 'Yellow'][player-1]
                player_info.append(f"Player {player} ({color_name}): Vertices {vertices}, Quality: {quality:.2f}")
        title += "  |  ".join(player_info)
    
    ax.set_title(title, fontsize=9, pad=15)
    
    # Set reasonable axis limits to show all tiles
    all_x = [p[0] for p in tile_centers.values()]
    all_y = [p[1] for p in tile_centers.values()]
    if all_x and all_y:
        margin = hex_radius * 1.5
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    
    # Add legend
    legend_elements = []
    for resource, color in resource_colors.items():
        legend_elements.append(mpatches.Patch(facecolor=color, edgecolor='black',
                                            label=f'{resource_abbrev[resource]}: {resource}'))
    
    if state:
        for player in sorted(state.houses.keys()):
            color = player_colors.get(player, '#000000')
            color_name = ['Red', 'Blue', 'Green', 'Yellow'][player-1]
            legend_elements.append(mpatches.Patch(facecolor=color, edgecolor='black',
                                                label=f'Player {player} ({color_name})'))
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()


def visualize_settlements_detailed(board: Board, state: State):
    """
    Create a detailed visualization showing which tiles each settlement touches.
    
    Args:
        board: Board object
        state: State object with player settlements
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    player_colors = {
        1: '#FF0000',  # Red
        2: '#0000FF',  # Blue
        3: '#00FF00',  # Green
        4: '#FFFF00',  # Yellow
    }
    
    resource_colors = {
        'wood': '#8B4513',
        'brick': '#CD5C5C',
        'wheat': '#FFD700',
        'ore': '#708090',
        'sheep': '#90EE90',
        'desert': '#F5DEB3'
    }
    
    for idx, player in enumerate(sorted(state.houses.keys())):
        if idx >= 4:
            break
        
        ax = axes[idx]
        ax.set_aspect('equal')
        ax.axis('off')
        
        vertices = state.houses[player]
        if len(vertices) != 2:
            continue
        
        color = player_colors.get(player, '#000000')
        color_name = ['Red', 'Blue', 'Green', 'Yellow'][player-1]
        
        # Draw all tiles
        tile_centers = {}
        for tile in board.tiles:
            row, col = tile['row'], tile['col']
            x, y = hex_to_pixel(row, col, size=1.2)
            tile_centers[tile['id']] = (x, y)
            
            resource = tile['resource']
            tile_color = resource_colors.get(resource, '#CCCCCC')
            
            # Check if this tile is touched by player's settlements
            is_touched = False
            for vertex in vertices:
                if tile['id'] in board.tiles_touching.get(vertex, []):
                    is_touched = True
                    break
            
            # Draw hexagon (highlighted if touched) - rotated 30 degrees
            hexagon = RegularPolygon((x, y), numVertices=6, radius=1.2,
                                    orientation=np.pi/6 + np.pi/6,
                                    facecolor=tile_color if not is_touched else '#FFD700',
                                    edgecolor='red' if is_touched else 'black',
                                    linewidth=3 if is_touched else 1,
                                    alpha=0.8 if is_touched else 0.5)
            ax.add_patch(hexagon)
            
            # Add number
            number = tile['number']
            if number is not None:
                ax.text(x, y, str(number), ha='center', va='center',
                       fontsize=9, fontweight='bold')
        
        # Draw settlements
        for vertex in vertices:
            tile_ids = board.tiles_touching.get(vertex, [])
            if tile_ids:
                positions = [tile_centers[tid] for tid in tile_ids if tid in tile_centers]
                if positions:
                    vx = sum(p[0] for p in positions) / len(positions)
                    vy = sum(p[1] for p in positions) / len(positions)
                    
                    settlement = Circle((vx, vy), 0.2, facecolor=color,
                                      edgecolor='black', linewidth=3, zorder=10)
                    ax.add_patch(settlement)
                    ax.text(vx, vy, str(player), ha='center', va='center',
                           fontsize=12, fontweight='bold', color='white', zorder=11)
        
        quality = state.quality_of_player(player)
        ax.set_title(f'Player {player} ({color_name})\n'
                    f'Settlements at vertices {vertices}\n'
                    f'Quality: {quality:.4f}', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

