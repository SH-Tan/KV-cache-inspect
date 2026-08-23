function count() {
  const lines = lines || [];
  const linesLength = lines.length;

  let count = 0;

  // Iterate through each pair of consecutive horizontal lines.
  for (let i = 0; i < lines.length - 1; i++) {
    const top = lines[i];
    const bottom = lines[i + 1];

    // For each row between top and bottom, count the number of vertical lines.
    let verticalCount = 0;
    for (let row = i + 1; row < lines.length; row++) {
      const currentRow = lines[row];
      const verticalCount += currentRow.length - 1; // Each line is a vertical line.
    }

    count += verticalCount;
  }

  return count;
}