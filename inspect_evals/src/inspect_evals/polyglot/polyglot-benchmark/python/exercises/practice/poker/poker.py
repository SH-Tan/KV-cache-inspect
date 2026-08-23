def best_hands(hands):
    card_map = {
        '2': 2,
        '3': 3,
        '4': 4,
        '5': 5,
        '6': 6,
        '7': 7,
        '8': 8,
        '9': 9,
        '10': 10,
        'J': 11,
        'Q': 12,
        'K': 13,
        'A': 14
    }
    
    def process_hand(hand):
        cards = hand.split()
        values = []
        suits = []
        for card in cards:
            if card in card_map:
                values.append(card_map[card])
            else:
                values.append(card)
        return values, suits
    
    def get_hand_info(values):
        # Returns the rank and highest card of the hand
        # Rank is higher for better hands
        # Highest card is the highest in the hand
        if len(set(values)) == 1 and values[0] - values[4] == 4:
            return ('straight flush', max(values))
        
        counts = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1
        four_of_a_kind = any(v == 4 for v in counts.values())
        if four_of_a_kind:
            return ('four_of_a_kind', max(values))
        
        counts = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1
        if sorted(counts.values(), reverse=True) == [3, 2]:
            return ('full house', max(values))
        
        if len(set(values)) == 1:
            return ('flush', max(values))
        
        if sorted(values) == [10, 11, 12, 13, 14] or sorted(values) == [14, 13, 12, 11, 10]:
            return ('straight', max(values))
        
        counts = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1
        three_of_a_kind = any(v == 3 for v in counts.values())
        if three_of_a_kind:
            return ('three_of_a_kind', max(values))
        
        pairs = [k for k, v in counts.items() if v == 2]
        if len(pairs) == 2:
            return ('two_pair', max(pairs))
        
        if len(pairs) == 1:
            return ('one_pair', max(pairs))
        
        return ('high_card', min(values))
    
    best_hand = None
    best_rank = -1
    best_highest = None
    
    for hand in hands:
        values, suits = process_hand(hand)
        info = get_hand_info(values)
        if best_hand is None or (info[0] > best_rank) or (info[0] == best_rank and info[1] > best_highest):
            best_hand = hand
            best_rank = info[0]
            best_highest = info[1]
    
    return best_hand