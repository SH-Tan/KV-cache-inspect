function transpose() {
    let rows = input.split('\n');
    if (rows.length === 0) return '';

    // Find the maximum number of columns
    let max_col = Math.max(...rows.map(row => row.length));

    // Pad each row with spaces to the left to make all rows have max_col length
    let paddedRows = rows.map(row => {
        let padded = row;
        while (padded.length < max_col) {
            padded += ' ';
        }
        return padded;
    });

    // Transpose the matrix
    let transposed = [];
    for (let col = 0; col < max_col; col++) {
        let newRow = [];
        for (let row of paddedRows) {
            newRow.push(row[col]);
        }
        transposed.push(newRow);
    }

    // Join each row into a string and then join all rows with newlines
    let result = transposed.map(row => row.join('')).join('\n');
    return result;
}