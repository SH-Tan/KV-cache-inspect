from itertools import permutations

# Define the possible values for each attribute
color = ["Red", "Green", "Ivory", "Yellow", "Blue"]
nationality = ["Englishman", "Spaniard", "Ukrainian", "Norwegian", "Japanese"]
pet = ["Dog", "Snails", "Fox", "Horse", "Zebra"]
drink = ["Tea", "Coffee", "Milk", "OrangeJuice", "Water"]
hobby = ["OldGold", "Kools", "Chesterfields", "LuckyStrike", "Parliaments"]

def solve_zebra_puzzle():
    # Iterate through all possible permutations for each attribute
    for colors in permutations(color):
        for nationality in permutations(nationality):
            for pets in permutations(pet):
                for drinks in permutations(drink):
                    for hobbies in permutations(hobby):
                        # Check all constraints
                        # Constraint 1: Norwegian lives in first house
                        if nationality[0] != "Norwegian":
                            continue
                        # Constraint 2: Englishman lives in red house
                        if colors[0] != "Red":
                            continue
                        # Constraint 3: Spaniard owns dog
                        if nationality.index("Spaniard") != pets.index("Dog"):
                            continue
                        # Constraint 4: Person in green drinks coffee
                        if colors[1] != "Green" and drinks[colors.index("Green")] == "Coffee":
                            continue
                        # Constraint 5: Person in yellow drinks milk
                        if colors[3] != "Yellow" and drinks[colors.index("Yellow")] == "Milk":
                            continue
                        # Constraint 6: Person in green is right of ivory
                        if colors.index("Green") - colors.index("Ivory") != 1:
                            continue
                        # Constraint 7: Snail owner likes dancing
                        if nationalities.index("Snails") != hobbies.index("Dance") or nationalities.index("Snails") != 0:
                            continue
                        # Constraint 8: Person in yellow is painter
                        if nationalities[colors.index("Yellow")] != "Painter":
                            continue
                        # Constraint 9: Person in middle drinks milk (but earlier constraints may have covered this)
                        # Constraint 10: Norwegian in first and next to blue house
                        if colors[0] != "Blue":
                            continue
                        # Constraint 11: Person who enjoys reading next to fox owner
                        if nationalities.index("Reading") != pets.index("Fox") -1:
                            continue
                        # Constraint 12: Painter's house next to horse
                        if nationalities.index("Horse") != nationalities.index("Paint") +1:
                            continue
                        # Constraint 13: Player who plays football drinks water
                        if drinks[nationalities.index("Football")] != "Water":
                            continue
                        # Constraint 14: Japanese person plays chess
                        if nationalities.index("Japanese") != hobbies.index("Chest") or nationalities.index("Japanese") != "ChessPlayer":
                            continue
                        # Constraint 15: Norwegian next to blue
                        if colors[0] != "Blue" or colors[4] != "Red":
                            continue
                        # Constraint 16: Person who plays chess drinks orange juice
                        if nationalities.index("ChessPlayer") != drinks.index("OrangeJuice"):
                            continue
                        # Constraint 17: Person who likes reading drinks tea
                        if nationalities.index("Reading") != drinks.index("Tea"):
                            continue
                        # Constraint 18: Person who owns zebra is next to dog owner
                        if nationalities.index("Zebra") != pets.index("Dog") +1:
                            continue
                        # Check all constraints passed
                        solution = {
                            'water_drinker': nationality[drinks.index("Water")],
                            'zebra_owner': nationalities[petes.index("Zebra")]
                        }
                        return solution
    # If no solution found (though problem states one exists)
    return None

water_drinker, zebra_owner = solve_zebra_puzzle()
print(f"The person who drinks water is: {water_drinker}")
print(f"The person who owns the zebra is: {zebra_owner}")