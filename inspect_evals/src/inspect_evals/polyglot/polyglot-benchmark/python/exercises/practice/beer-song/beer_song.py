def recite(start, take=1):
    current = start
    while current >= 1:
        bottles = f"{current} bottles of beer on the wall, {current} bottles of beer."
        next_bottles = current - 1
        if next_bottles == 0:
            next_line = "Go to the store and buy some more, 99 bottles of beer on the wall."
        else:
            next_line = f"{next_bottles} bottle(s) of beer on the wall, {next_bottles} bottle(s) of beer."
        bottles += next_line
        print(bottles)
        current = next_bottles
    if start != 0:
        current = start - 1
        while current >= 1:
            bottles = f"{current} bottle(s) of beer on the wall, {current} bottle(s) of beer."
            next_bottles = current - 1
            if next_bottles == 0:
                next_line = "Go to the store and buy some more, 99 bottles of beer on the wall."
            else:
                next_line = f"{next_bottles} bottle(s) of beer on the wall, {next_bottles} bottle(s) of beer."
            bottles += next_line
            print(bottles)
            current = next_bottles