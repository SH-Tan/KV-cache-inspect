def scale(tonic, interval):
    chromatic_scale = ['A', 'A♯', 'B', 'C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯']
    
    # Convert intervals to step sizes
    steps = []
    for c in interval:
        if c == 'M':
            steps.append(2)
        elif c == 'm':
            steps.append(1)
        elif c == 'd':
            steps.append(1)
        elif c == 'b':
            steps.append(1)
    
    # Generate the scale
    current_note = tonic
    scale_notes = [current_note]
    for step in steps:
        index = chromatic_scale.index(current_note)
        new_index = (index + step) % 12
        new_note = chromatic_scale[new_index]
        scale_notes.append(new_note)
        current_note = new_note
    
    return scale_notes