import random

def mazeGenerator(rows, cols):
    # Orthogonal perfect maze.
    # Maximum depth of this is the number of cells.
    directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    # Create adjacency set of walls (walls everywhere).
    # We have also have walls between, for example, (0, 0) and (0, -1).
    walls = list()
    for row in range(rows):
        for col in range(cols):
            if row == 0:
                walls.append({(row, col), (row - 1, col)})
            if col == 0:
                walls.append({(row, col), (row, col - 1)})
            walls.append({(row, col), (row + 1, col)})
            walls.append({(row, col), (row, col + 1)})
    visited = set()
    def removeWalls(row, col):
        visited.add((row, col))
        # In a random order of directions,
        for drow, dcol in random.sample(directions, len(directions)):
            nrow, ncol = row + drow, col + dcol
            # If new cell is in the maze and not visited,
            if ((0 <= nrow < rows and 0 <= ncol < cols) and
                ((nrow, ncol) not in visited)):
                # Then break down the wall!
                walls.remove({(row, col), (nrow, ncol)})
                # Call removeWalls on the new cell.
                removeWalls(nrow, ncol)
        # Base case: dead end.
    removeWalls(0, 0)
    return walls
