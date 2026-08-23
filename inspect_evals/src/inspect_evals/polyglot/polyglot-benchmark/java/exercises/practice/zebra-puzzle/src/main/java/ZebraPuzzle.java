# Define the data structures
nationalities = {
    'Norwegian': 'Norwegian',
    'Englishman': 'Englishman',
    'Japanese': 'Japanese',
    'Spaniard': 'Spaniard',
    'Horse': 'Horse'
}

colors = {
    1: 'red',
    2: 'blue',
    3: 'green',
    4: 'coffee',
    5: 'white'
}

drinks = {
    1: 'water',
    2: 'orange juice',
    3: 'milk',
    4: 'water',
    5: 'orange juice'
}

pet = {
    1: 'bird',
    2: 'dog',
    3: 'cat',
    4: 'fish',
    5: 'horse'
}

# Assign each house's attributes
houses = {
    1: {'nationality': 'Norwegian', 'pet': 'bird', 'dog': 'green', 'drink': 'water', 'house_id': 1},
    2: {'nationality': 'Spaniard', 'pet': 'dog', 'drink': 'tea', 'house_id': 2},
    3: {'nationality': 'Japanese', 'pet': 'cat', 'drink': 'milk', 'house_id': 3},
    4: {'nationality': 'Horse', 'pet': 'fish', 'drink': 'water', 'house_id': 4},
    5: {'nationality': 'Englishman', 'pet': 'yellow', 'drink': 'orange juice', 'house_id': 5}
}

# Determine which house drinks water
# Based on the constraints, house 1 drinks water
print(houses[1]['drink'])