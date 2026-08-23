export class Board {
  constructor() {
    this.grid = [];
    const rows = this.toString().split('\n');
    for (const row of rows) {
      this.grid.push(row.split(''));
    }
  }

  winner() {
    // Check if O has a path to the right
    const oPath = false;
    const rows = this.grid.length;
    if (rows === 0) return 'Tie';
    const cols = this.grid[0].length;

    // Check if starting cell is 'O'
    if (this.grid[0][0] !== 'O') return 'Tie';
    if (this.grid[0][0] !== '.' && this.grid[0][0] !== 'X') return 'Tie';

    // BFS for O's path
    const visited = new Array(rows).fill(new Array(cols)).fill(false);
    const queue = [[0, 0]];
    visited[0][0] = true;

    while (queue.length > 0) {
      const [i, j] = queue.shift();
      const targetCol = cols - 1;

      if (j === targetCol) {
        oPath = true;
        break;
      }

      for (let di = -1; di <= 1; di++) {
        for (let dj = -1; dj <= 1; dj++) {
          if (di === 0 && dj === 0) continue;
          const newRow = i + di;
          const newCol = j + dj;

          if (newRow >= 0 && newRow < rows) {
            if (newCol >= 0 && newCol < this.grid[newRow].length) {
              if (this.grid[newRow][newCol] === 'O' && !visited[newRow][newCol]) {
                visited[newRow][newCol] = true;
                queue.push([newRow, newCol]);
              }
            }
          }
        }
      }
    }

    if (oPath) return 'O';

    // Check if X has a path to the bottom
    const xPath = false;
    if (this.grid.length > 0) {
      const startRow = 0;
      const startCol = 0;
      if (this.grid[startRow][startCol] !== 'X') return 'Tie';
      if (this.grid[startRow][startCol] !== '.' && this.grid[startRow][startCol] !== 'X') return 'Tie';

      const visited = new Array(this.grid.length).fill(new Array(this.grid[0].length)).fill(false);
      const queue = [[startRow, startCol]];
      visited[startRow][startCol] = true;

      while (queue.length > 0) {
        const [i, j] = queue.shift();
        const targetRow = this.grid.length - 1;

        if (i === targetRow) {
          xPath = true;
          break;
        }

        for (let di = -1; di <= 1; di++) {
          for (let dj = -1; dj <= 1; dj++) {
            if (di === 0 && dj === 0) continue;
            const newRow = i + di;
            const newCol = j + dj;

            if (newRow >= 0 && newRow < this.grid.length) {
              if (newCol >= 0 && newCol < this.grid[newRow].length) {
                if (this.grid[newRow][newCol] === 'X' && !visited[newRow][newCol]) {
                  visited[newRow][newCol] = true;
                  queue.push([newRow, newCol]);
                }
              }
            }
          }
        }
      }
    }

    if (xPath) return 'X';

    return 'Tie';
  }
}