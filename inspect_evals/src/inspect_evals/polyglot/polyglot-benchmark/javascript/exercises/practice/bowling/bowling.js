class Bowling:
    def __init__(self):
        self.rolls = []
        self.current_score = 0

    def roll(self, pins):
        self.rolls.append(pins)

    def score(self):
        total = 0
        for frame in range(1, 11):
            if frame == 10:
                # Handle the 10th frame differently
                if len(self.rolls) >= 2 and self.rolls[0] == 10:
                    # Strike, add next two rolls
                    total += 10
                    total += self.rolls[1] + self.rolls[2]
                elif len(self.rolls) >= 2 and self.rolls[0] + self.rolls[1] == 10:
                    # Spare, add next roll
                    total += 10
                    total += self.rolls[2]
                else:
                    # Open frame, add sum of two rolls
                    total += self.rolls[0] + self.rolls[1]
            else:
                # Process other frames
                if len(self.rolls) == 0:
                    # Initial state, perhaps frame 1 is a strike
                    self.rolls.append(pins)
                else:
                    first_roll = self.rolls[0]
                    if first_roll == 10:
                        # Strike, add next two rolls
                        total += 10
                        total += self.rolls[1] + self.rolls[2]
                    elif first_roll + self.rolls[1] == 10:
                        # Spare, add next roll
                        total += 10
                        total += self.rolls[2]
                    else:
                        # Open frame, sum of two rolls
                        total += self.rolls[0] + self.rolls[1]
        return total