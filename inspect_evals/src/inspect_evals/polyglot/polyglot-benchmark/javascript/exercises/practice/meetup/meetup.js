function findTe nthDate(m, y, pos) {
    // Determine the first day of the given month
    const firstDay = new Date(y, m - 1, 0).getDate();
    
    // Generate all te nth days in the month
    const allTe nthDays = [];
    for (let i = 13; i <= 19; i++) {
        const day = new Date(firstDay, 0, i);
        if (day.getDate() <= lastDay(m, y)) {
            allTe nthDays.push(day);
        }
    }
    
    // Return the position-th day
    return allTe nthDays[pos - 1];
}

// Helper function to get the last day of the month
function lastDay(m, y) {
    const firstDay = new Date(y, m - 1, 0).getDate();
    const daysInMonth = new Date(y, m - 1, 31).getDate();
    return daysInMonth - firstDay + 1;
}