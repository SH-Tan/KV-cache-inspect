from collections import Counter

def score(dice, category):
    counts = Counter(dice)
    
    # Check for each category in order
    
    # Ones
    if all(d == 1 for d in dice):
        return counts[1] * 1
    
    # Twos
    elif all(d == 2 for d in dice):
        return counts[2] * 2
    
    # Threes
    elif all(d == 3 for d in dice):
        return counts[3] * 3
    
    # Fours
    elif all(d == 4 for d in dice):
        return counts[4] * 4
    
    # Fives
    elif all(d == 5 for d in dice):
        return counts[5] * 5
    
    # Sixes
    elif all(d == 6 for d in dice):
        return counts[6] * 6
    
    # Full House
    elif len(counts) == 2 and (list(counts.values()) == [2, 3]):
        return sum(dice)
    
    # Four of a Kind
    elif any(v == 4 for v in counts.values()):
        for num, cnt in counts.items():
            if cnt == 4:
                return num * 4
    
    # Little Straight
    elif sorted(dice) in [[1,2,3,4,5], [2,3,4,5,6]]:
        return sum(dice)
    
    # Big Straight
    elif {2,3,4,5,6}.issubset(dice):
        return sum(dice)
    
    # If none match, return 0
    return 0