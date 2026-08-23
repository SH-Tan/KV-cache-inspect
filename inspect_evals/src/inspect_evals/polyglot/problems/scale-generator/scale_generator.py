<<<<<<< Updated upstream
Your code
=======
class Scale:
    def __init__(self, tonic):
        self.tonic = tonic
    
    def chromatic(self):
        chromatic_notes = ['A', 'A♯', 'B', 'C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯']
        index = chromatic_notes.index(self.tonic)
        return chromatic_notes[index : index + 12]
    
    def intervals(self, intervals):
        chromatic_notes = ['A', 'A♯', 'B', 'C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯']
        index = chromatic_notes.index(self.tonic)
        result = [self.tonic]
        for interval in intervals:
            step = 2 if interval == 'M' else 1
            new_index = (index + step) % 12
            new_note = chromatic_notes[new_index]
            result.append(new_note)
        return result
>>>>>>> Stashed changes
