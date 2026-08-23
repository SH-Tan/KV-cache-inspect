export class Triangle {
  constructor(...sides) {
    throw new Error('Remove this statement and implement this function');
  }

  get isEquilateral() {
    throw new Error('Remove this statement and implement this function');
  }

  get isIsosceles() {
    throw new Error('Remove this statement and implement this function');
  }

  get isScalene() {
    throw new Error('Remove this statement and implement this function');
  }
}

// Implementing the functions as methods of the Triangle class
function isEquilateral(sides) {
    return sides[0] === sides[1] && sides[1] === sides[2];
}

function isIsosceles(sides) {
    return sides[0] === sides[1] || sides[1] === sides[2] || sides[0] === sides[2];
}

function isScalene(sides) {
    return !isEquilateral(sides) && !isIsosceles(sides);
}