function chromatic(tonic) {
    const chromaticOrder = ['C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B'];
    const scale = [tonic];
    let index = chromaticOrder.indexOf(tonic);
    for (let i = 1; i < 12; i++) {
        let note = chromaticOrder[index + i];
        scale.push(note);
        index += 1;
    }
    return scale;
}

function interval(intervals) {
    const intervalMap = {
        'b': -1,
        'B♭': -2,
        'bflat': -2,
        'C♯': +1,
        'd': +1,
        'E': +1,
        'F': +1,
        'F♯': +1,
        'G': +1,
        'G♯': +1,
        'A': +1,
        'A♯': +1,
        'B': +1,
        'B♭': -1,
        'C': +1
    };

    const scale = [tonic];
    let semitone = 0;
    for (const interval of intervals) {
        let step = intervalMap[interval];
        semitone += step;
        const note = chromatic(semitone).toUpperCase();
        scale.push(note);
    }
    return scale;
}