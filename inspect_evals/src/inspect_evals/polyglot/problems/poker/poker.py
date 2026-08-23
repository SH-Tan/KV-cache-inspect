<<<<<<< Updated upstream
transformers==2.16.1
nltk==3.8.5
scikit-learn==0.25.1
pandas==1.1.4
numpy==1.20.2
termcolor==4.1
psycopg2-binary==2.8.7
dbt
jinja2==3.1.2
adafruit-circuitpython-stepper==6.5.1
adafruit-circuitpython-ssd1346==6.1.1
adafruit-circuitpython-displayio==0.3.4
adafruit-circuitpython-ft232h==4.2.1
adafruit-circuitpython-ads1x15==4.1.5
adafruit-circuitpython-si7006==6.0
adafruit-circuitpython-keyon==1.4.0
adafruit-circuitpython-batt-cli==0.4.0
'''
SimplyIndex
Your first answer
=======
def best_hands(hands):
    def evaluate_hand(hand):
        # Convert each card to numerical value and suit
        cards = []
        for card in hand:
            rank = card[0]
            suit = card[1]
            if rank == 'T':
                num = 10
            elif rank == 'J':
                num = 11
            elif rank == 'Q':
                num = 12
            elif rank == 'K':
                num = 13
            elif rank == 'A':
                num = 14
            else:
                num = int(rank)
            cards.append((num, suit))
        
        # Check for flush
        suits = [c[1] for c in cards]
        flush = len(set(suits)) == 1
        
        # Check for straight
        sorted_nums = sorted([c[0] for c in cards])
        straight = False
        if len(set(sorted_nums)) == 5:  # All unique
            straight = True
        elif sorted_nums == [10, 2, 3, 4, 5]:  # Wheel
            straight = True
        straight = True if straight else False
        
        # Create a frequency dictionary
        counts = {}
        for num in sorted_nums:
            counts[num] = counts.get(num, 0) + 1
        counts = sorted(counts.values(), reverse=True)
        
        # Evaluate the hand
        if flush and straight:
            return (0, 1)  # Straight flush
        if flush:
            return (0, 2)  # Flush
        
        if counts[0] == 4:
            return (0, 3)  # Four of a kind
        
        if counts == [3, 2]:
            return (0, 4)  # Full house
        
        if flush:
            return (0, 2)  # Redundant check for flush
        
        if straight:
            return (0, 5)  # Straight
        
        if counts[0] == 3:
            return (0, 6)  # Three of a kind
        
        if counts[0] == 2 and counts[1] == 2:
            return (0, 7)  # Two pair
        
        if counts[0] == 2:
            return (0, 8)  # One pair
        
        # If none, return high card
        return (0, 9)
    
    evaluated = []
    for hand in hands:
        strength, rank = evaluate_hand(hand)
        evaluated.append((strength, rank))
    
    # Find the maximum evaluated hand
    max_strength, max_hand = max(evaluated, key=lambda x: x[0])
    return max_hand
>>>>>>> Stashed changes
