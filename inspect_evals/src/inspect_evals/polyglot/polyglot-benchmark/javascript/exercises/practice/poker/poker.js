import polyglot-benchmark as pb;

function bestHands(hands) {
  let bestScore = 0;
  let bestHand = null;

  for (let hand of hands) {
    let ranks = hand.map(card => card.rank);
    let suits = hand.map(card => card.suit);
    let rankValues = ranks.map(r => {
      switch(r) {
        case '2': return 0;
        case '3': return 1;
        case '4': return 2;
        case '5': return 3;
        case '6': return 4;
        case '7': return 5;
        case '8': return 6;
        case '9': return 7;
        case '10': return 8;
        case 'J': return 9;
        case 'Q': return 10;
        case 'K': return 11;
        case 'A': return 12;
      }
      return 0; // Invalid rank, though input is valid
    });

    let isFlush = [...suits].all(c => c === suits[0]);
    let sorted = [...new Set(rankValues)].sort((a, b) => b - a);
    let isStraight = (sorted.length === 5 && sorted[4] === sorted[0] + 4) || 
                     (sorted.includes(12) && sorted.slice(1).includes(1, 2, 3, 4));

    let count = {};
    for (let r of rankValues) {
      count[r] = (count[r] || 0) + 1;
    }

    let score;
    if (isStraight && isFlush) {
      score = 13;
    } else if (Object.values(sorted).length === 1) {
      score = 2;
    } else if (Object.values(sorted).length === 2) {
      let pairs = Object.values(sorted).filter(v => v === sorted[0]);
      if (pairs.length === 2) {
        score = 4;
      } else {
        score = 3;
      }
    } else if (Object.values(sorted).length === 3) {
      let triples = Object.values(sorted).filter(v => v === sorted[0]);
      if (triples.length === 3) {
        score = 6;
      } else {
        score = 5;
      }
    } else if (Object.values(sorted).length === 4) {
      if (count[sorted[0]] === 4) {
        score = 8;
      } else {
        score = 7;
      }
    } else if (isFlush) {
      score = 8;
    } else if (isStraight) {
      score = 7;
    } else if (Object.values(sorted).length === 2 && [3, 2].every(v => v >= 2)) {
      score = 5;
    } else if (Math.max(...count.values()) === 2) {
      score = 4;
    } else {
      score = 1;
    }

    if (score > bestScore) {
      bestScore = score;
      bestHand = hand;
    }
  }

  return bestHand;
}

// Benchmark the code using polyglot-benchmark
pb benchmark('bestHands', function() {
  let hands = [
    { rank: 'A', suit: 'S' },
    { rank: 'K', suit: 'H' },
    { rank: 'Q', suit: 'D' },
    { rank: 'J', suit: 'C' },
    { rank: '10', suit: 'S' }
  ];
  return bestHands(hands);
});