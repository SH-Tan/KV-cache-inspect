class QueenAttack {
  constructor({
    white: [whiteRow, whiteColumn],
    black: [blackRow, blackColumn]
  }) {
    this.x1 = white[0].charCodeAt(0) - 'a'.charCodeAt(0);
    this.y1 = white[1] - 1;
    this.x2 = black[0].charCodeAt(0) - 'a'.charCodeAt(0);
    this.y2 = black[1] - 1;

    this.x1 = this.x1;
    this.y1 = this.y1;
    this.x2 = this.x2;
    this.y2 = this.y2;
  }

  public function canAttack() {
    const sameRow = this.y1 === this.y2;
    const sameColumn = this.x1 === this.x2;
    const sameDiagonal = Math.abs(this.x1 - this.x2) === Math.abs(this.y1 - this.y2);

    return sameRow || sameColumn || sameDiagonal;
  }

  // toString method is not implemented as per problem statement
}