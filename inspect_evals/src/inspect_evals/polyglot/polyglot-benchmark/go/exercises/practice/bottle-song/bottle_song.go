def Recite(startBottles):
    lines = []
    for current_bottles in range(startBottles, 0, -1):
        lines.append(f"{current_bottles} green bottles hanging on the wall, same for the other.")
        lines.append(f"And if one green bottle should fall, there'll be {current_bottles - 1} green bottles hanging on the wall.")
    lines.append("One green bottle hanging on the wall, same for the other, and if one falls, no green bottles hanging on the wall.")
    return '\n'.join(lines)