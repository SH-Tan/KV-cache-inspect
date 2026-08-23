function tournamentTally(input) {
    let stats = {};

    const lines = input.split('\n');

    for (let line of lines) {
        line = line.trim();
        if (!line) continue;

        let parts = line.split(';');
        if (parts.length !== 3) continue;

        let team1 = parts[0].trim();
        let team2 = parts[1].trim();
        let result = parts[2].trim();

        if (result === 'win') {
            let winner = team1;
            let loser = team2;
        } else if (result === 'loss') {
            let winner = team2;
            let loser = team1;
        } else {
            let winner = team1;
            let loser = team2;
        }

        // Update winner's stats
        stats[winner]['W'] += 1;
        stats[winner]['P'] += 3;
        stats[loser]['L'] += 1;

        // Ensure default values for missing keys
        if (!stats[winner].has('W')) stats[winner]['W'] = 0;
        if (!stats[winner].has('D')) stats[winner]['D'] = 0;
        if (!stats[winner].has('L')) stats[winner]['L'] = 0;
        if (!stats[winner].has('P')) stats[winner]['P'] = 0;
    }

    // Collect and sort the teams alphabetically
    const teams = Object.keys(stats).sort();

    for (let team of teams) {
        let w = stats[team].get('W') || 0;
        let d = stats[team].get('D') || 0;
        let l = stats[team].get('L') || 0;
        let p = stats[team].get('P') || 0;

        console.log(`${team} | ${w} | ${d} | ${l} | ${p}`);
    }
}