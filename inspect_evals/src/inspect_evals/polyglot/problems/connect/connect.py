<<<<<<< Updated upstream
=======
class ConnectGame:
    def __init__(self, board):
        self.top_row = board[0]
        self.bottom_row = board[-1]
        self.left_col = [row[0] for row in board]
        self.right_col = [row[-1] for row in board]
        
        self.o_connected = False
        self.x_connected = False

        def can_win(player, top_row, bottom_row, left_col, right_col):
            # Check if player can connect top and bottom
            if all(cell == player for cell in top_row) and all(cell == player for cell in bottom_row):
                visited = set()
                from collections import deque
                queue = deque()
                queue.append((0, 0))
                visited.add((0, 0))
                found = False
                while queue:
                    r, c = queue.popleft()
                    if r == len(board) - 1:
                        found = True
                        break
                    for dr, dc in [(-1, 0), (1, 0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < len(board) and 0 <= nc < len(board[0]) and board[nr][nc] == player and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                return found

            # Check if player can connect left and right
            if all(cell == player for cell in left_col) and all(cell == player for cell in right_col):
                visited = set()
                from collections import deque
                queue = deque()
                queue.append((0, 0))
                visited.add((0, 0))
                found = False
                while queue:
                    r, c = queue.popleft()
                    if c == len(board[0]) - 1:
                        found = True
                        break
                    for dr, dc in [(0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < len(board) and 0 <= nc < len(board[0]) and board[nr][nc] == player and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                return found

            # Check if all left and right are player
            if all(cell == player for cell in left_col) and all(cell == player for cell in right_col):
                return True

            return False

        if self.top_row and self.bottom_row:
            self.o_connected = can_win('O', self.top_row, self.bottom_row, self.left_col, self.right_col)
        if self.top_row and self.bottom_row:
            self.x_connected = can_win('X', self.top_row, self.bottom_row, self.left_col, self.right_col)

    def __repr__(self):
        return "O wins" if self.o_connected else "X wins" if self.x_connected else "draw"
>>>>>>> Stashed changes
