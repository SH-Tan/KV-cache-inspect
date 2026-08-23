# Define all the gifts in order
gifts = [
    "a Partridge in a Pear Tree",
    "two Turtle Doves",
    "three French Hens",
    "four Calling Birds",
    "five Gold Rings",
    "six Geese-a-Laying",
    "seven Swans-a-Swimming",
    "eight Maids-a-Milking",
    "nine Ladies Dancing",
    "ten Lords-a-Leaping",
    "eleven Pipers Piping",
    "twelve Drummers-Da-Laying"
]

# Initialize the total string
total = ""

# Loop through each day from 1 to 12
for day in range(1, 13):
    verse = f"{day}"
    # Append each gift with its count
    for i in range(day):
        if i >= len(gifts):
            break
        gift = gifts[i]
        verse += f" {i+1} {gift}"
    # Add the verse to the total
    total += verse

# Print the total string
print(total)