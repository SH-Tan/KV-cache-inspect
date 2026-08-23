def phone_number(text):
    # Extract all digits from the input string
    digits = ''.join([c for c in text if c.isdigit()])
    
    # The NANP format is 10 digits: 3 exchange code + 3 local + 4 country code
    # We take the first three digits as the exchange code, next three as local, and last four as country
    # However, to ensure the correct format, we process the digits as follows:
    # The first three digits are the exchange code, next three are the local part, and the last four are the country code.
    # Thus, the code can be structured to return the digits as the final cleaned phone number.
    return digits