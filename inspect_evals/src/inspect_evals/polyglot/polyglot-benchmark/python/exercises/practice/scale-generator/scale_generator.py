class Scale:
    def __init__(self, tonic):
        self.tonic = tonic

    def chromatic(self):
        chromatic_scale = ['C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B']
        return chromatic_scale

    def interval(self, intervals):
        note_to_semitone = {
            'C': 0,
            'C♯': 1,
            'D': 2,
            'D♯': 3,
            'E': 4,
            'F': 5,
            'F♯': 6,
            'G': 7,
            'G♯': 8,
            'A': 9,
            'A♯': 10,
            'B': 11
        }
        intervals_map = {
            'm': 1,
            'M': 2,
            'a': 1,
            'd': 2
        }
        current_note = self.tonic
        current_semitone = note_to_semitone.get(current_note, 0)
        notes = [current_note]
        for interval in intervals:
            step = intervals_map.get(interval, 0)
            current_semitone += step
            current_note = chromatic_scale[current_semitone % len(chromatic_scale)]
            notes.append(current_note)
        return notes

    def scale(self):
        major_scale = []
        current_note = self.tonic
        current_semitone = note_to_semitone.get(current_note, 0)
        major_scale.append(current_note)
        for i in range(6):
            if i % 3 == 0:
                step = 2
            else:
                step = 2
            current_semitone += step
            current_note = chromatic_scale[current_semitone % len(chromatic_scale)]
            major_scale.append(current_note)
        return major_scale