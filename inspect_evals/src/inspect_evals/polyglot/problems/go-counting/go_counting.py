<<<<<<< Updated upstream
=======
from collections import deque

class Board:
    """Count territories of each player in a Go board

    Args:
        board (list[str]): A two-dimensional Go board
    """

    def __init__(self, board):
        self.grid = []
        for row in board:
            self.grid.append(row[1:-1])

    def territories(self):
        """Find the owner and the territories of the whole board

        Args:
            none

        Returns:
            dict(str, set): A dictionary whose key being the owner
                            (i.e. "W", "B", "".  The value being a set
                            of coordinates owned by the owner.
        """
        visited = [[False for _ in row] for row in self.grid]
        owner_set = set()

        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                if self.grid[i][j] == 'I' and not visited[i][j]:
                    queue = deque()
                    queue.append((i, j))
                    visited[i][j] = True
                    connected_component = []
                    has_o = False

                    while queue:
                        x, y = queue.popleft()
                        if self.grid[x][y] == 'O':
                            has_o = True
                            break
                        connected_component.append((x, y))
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx = x + dx
                            ny = y + dy
                            if 0 <= nx < len(self.grid) and 0 <= ny < len(self.grid[nx]):
                                if not visited[nx][ny] and self.grid[nx][ny] == 'I':
                                    visited[nx][ny] = True
                                    queue.append((nx, ny))

                    if not has_o:
                        for coord in connected_component:
                            owner_set.add(coord)

        if len(owner_set) == 0:
            self.owner = 'W'
        elif len(owner_set) == len(self.grid):
            self.owner = 'B'
        else:
            if len(owner_set) > len(self.owner_set):
                self.owner = 'B'
            else:
                self.owner = 'W'

    def territory(self, x, y):
        """Check if the given coordinate is part of the owner's territory

        Args:
            x (int): Row index
            y (int): Column index

        Returns:
            bool: True if the coordinate is part of the owner's territory, False otherwise
        """
        if x < 0 or x >= len(self.grid) or y < 0 or y >= len(self.grid[0]):
            return False

        if self.grid[x][y] != 'I':
            return False

        return (x, y) in self.owner_set
>>>>>>> Stashed changes
