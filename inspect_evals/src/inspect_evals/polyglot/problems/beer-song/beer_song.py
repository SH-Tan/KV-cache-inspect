<<<<<<< Updated upstream
=======
def recite(start, take=1):
    if start == 0:
        print("No more bottles of beer on the wall, no more bottles of beer.")
        print("Go to the store and buy some more, 99 bottles of beer on the wall.")
        return
    print(f"{start} bottles of beer on the wall, {start} bottles of beer.")
    print(f"Take one down and pass it around, {start-1} bottles of beer on the wall.")
    recite(start - 1, take)
>>>>>>> Stashed changes
