def bowling_game():
    rolls = []
    total = 0

    for frame in range(1, 11):
        if frame < 10:
            if frame == 1:
                first = rolls[0]
                if first == 10:
                    # Strike for frame 1
                    if len(rolls) >= 2:
                        total += 10
                        total += rolls[1]
                        if len(rolls) >= 3:
                            total += rolls[2]
                    # Add current roll to rolls
                    rolls.append(rolls[0])
                else:
                    if len(rolls) >= 2 and first + rolls[1] == 10:
                        # Spare for frame 1
                        total += 10
                        if len(rolls) >= 3:
                            total += rolls[2]
                    else:
                        total += first + rolls[1]
                    # Add current roll to rolls
                    rolls.append(rolls[0])
            else:
                first = rolls[0]
                if first == 10:
                    # Strike for frame > 1
                    if len(rolls) >= 2:
                        total += 10
                        total += rolls[1] if len(rolls) >= 2 else 0
                        if len(rolls) >= 3:
                            total += rolls[2] if len(rolls) >= 3 else 0
                    # Add current roll to rolls
                    rolls.append(rolls[0])
                else:
                    if len(rolls) >= 2 and first + rolls[1] == 10:
                        # Spare for frame > 1
                        total += 10
                        if len(rolls) >= 3:
                            total += rolls[2]
                    else:
                        total += first + rolls[1]
                    # Add current roll to rolls
                    rolls.append(rolls[0])
        else:
            # Process frame 10
            if len(rolls) == 0:
                # Invalid, but just in case
                pass
            else:
                first = rolls[0]
                if first == 10:
                    # Strike for frame 10
                    if len(rolls) >= 2:
                        total += 10
                        total += rolls[1]
                        if len(rolls) >= 3:
                            total += rolls[2]
                    # Add current roll to rolls
                    rolls.append(rolls[0])
                else:
                    if len(rolls) >= 2 and first + rolls[1] == 10:
                        # Spare for frame 10
                        total += 10
                        if len(rolls) >= 3:
                            total += rolls[2]
                    else:
                        total += first + rolls[1]
                    # Add current roll to rolls
                    rolls.append(rolls[0])
    
    return total