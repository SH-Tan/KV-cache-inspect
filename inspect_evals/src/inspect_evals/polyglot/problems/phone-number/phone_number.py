<<<<<<< Updated upstream
=======
import re

class PhoneNumber:
    def __init__(self, number):
        self.number = number
        
    def __str__(self):
        digits = re.findall(r'\d+', self.number)
        if len(digits) != 10:
            raise ValueError("Invalid phone number: incorrect number of digits")
        if digits[0] != '1':
            raise ValueError("Invalid country code")
        area_code = digits[1:4]
        exchange_code = digits[4:7]
        local_number = digits[7:10]
        if len(area_code) != 3 or len(exchange_code) != 3 or len(local_number) != 4:
            raise ValueError("Invalid area code, exchange code, or local number")
        cleaned = area_code + exchange_code + local_number
        return cleaned
>>>>>>> Stashed changes
