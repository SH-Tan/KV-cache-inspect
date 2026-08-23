<<<<<<< Updated upstream
def total(basket):
    # Определение дискаунта при покупке четных числа книг
    def get_discount_value(discount_type):
        if discount_type == 25:
            return 0.25
        elif discount_type == 10:
            return 0.10
        elif discount_type == 20:
            return 0.20
        elif discount_type == 10:
            return 0.10
        else:
            return 0.20

    # Сумма без учета дискаунта
    total_sum = sum_item_prices(copies)

    # Вычисление дискаунта на группу покупок книг
    group_discount = get_discount_value(discount_type)

    # Вычисление общей суммы с учетом дискаунта на группы
    total_sum -= total_sum *  group_discount

    # Возвращение стоимости книги
    return total_sum
=======
>>>>>>> Stashed changes
