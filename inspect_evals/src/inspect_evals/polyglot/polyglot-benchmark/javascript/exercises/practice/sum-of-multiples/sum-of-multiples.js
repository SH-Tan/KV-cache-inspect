export const sum = (level, baseValues = []) => {
  const multiples = new Set();
  for (const base of baseValues) {
    for (let i = base; i < level; i += base) {
      multiples.add(i);
    }
  }
  return Array.from(multiples).reduce((acc, curr) => acc + curr, 0);
};