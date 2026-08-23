class ConnectGame:
    def __init__(self, board):
        self.grid = []
        for line in board.split('\n'):
            row = list(line.strip())
            self.grid.append(row)
    
    def get_winner(self):
        current_player = 'O'
        for r in range(len(self.grid)):
            for c in range(len(self.grid[r])):
                if current_player == 'O':
                    if self.grid[r][c] == 'O':
                        self.grid[r][c] = 'O'
                    else:
                        self.grid[r][c] = 'X'
                else:
                    if self.grid[r][c] == 'X':
                        self.grid[r][c] = 'X'
                    else:
                        self.grid[r][c] = 'O'
                # Check if this move caused a win
                if self.is_win(self.grid, current_player, r, c):
                    return current_player
                current_player = 'X' if current_player == 'O' else 'O'
        return None

    def is_win(self, grid, player, r, c):
        # Check row
        row_win = True
        for col in range(len(grid[r])):
            if grid[r][col] != player:
                row_win = False
                break
        if row_win:
            return True
        
        # Check column
        col_win = True
        for row in range(len(grid)):
            if grid[row][c] != player:
                col_win = False
                break
        if col_win:
            return True
        
        return False