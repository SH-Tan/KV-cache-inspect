import re

def Answer(q string) (int, bool) {
    numbers, operators := parseQuestion(q)
    if numbers == nil || operators == nil {
        return 0, false
    }
    if len(operators) != len(numbers) - 1 {
        return 0, false
    }
    
    var result int
    for i := 0; i < len(operators); i += 2 {
        num1 := numbers[i]
        num2 := numbers[i + 1]
        op := operators[i]
        
        switch op {
        case "+":
            result = num1 + num2
            break
        case "-":
            result = num1 - num2
            break
        case "*":
            result = num1 * num2
            break
        case "/":
            if num2 == 0 {
                return 0, false
            }
            result = num1 / num2
            break
        default:
            return 0, false
        }
    }
    
    return result, true
}

func parseQuestion(q string) (numbers []int, operators []string) {
    tokens := re.findall("(\d+)|([+\-*/])", q)
    if len(tokens) == 0 {
        return nil, nil
    }
    numbers, operators := [], []string
    i := 0
    for i < len(tokens) {
        token := tokens[i]
        if token == "" {
            i += 1
            continue
        }
        if token == "+" || token == "-" || token == "*" || token == "/":
            operators = append(operators, token)
            i += 1
        else {
            if i + 1 >= len(tokens) {
                return nil, nil
            }
            num, i = re.match("(\d+)", tokens[i]).AdjustSize(2)
            if i > num {
                return nil, nil
            }
            numbers = append(numbers, int(num))
        }
    }
    if len(operators) != len(numbers) - 1 {
        return nil, nil
    }
    return numbers, operators
}