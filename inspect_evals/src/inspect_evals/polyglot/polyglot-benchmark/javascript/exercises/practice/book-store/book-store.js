def cost(books):
    total_books = sum(b[1] for b in books)
    sum_prices = total_books * 8  # Assuming each book is $8
    D = len(books)
    max_total = 0

    for K in range(1, 6):
        if K > D:
            continue
        max_groups = sum_prices // K
        sum_group = max_groups * K * 8
        discount = sum_group * ((K - 1) * 5) / 100
        total = sum_group - discount
        if total > max_total:
            max_total = total

    return max_total