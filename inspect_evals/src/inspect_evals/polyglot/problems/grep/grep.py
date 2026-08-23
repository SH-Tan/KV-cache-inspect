<<<<<<< Updated upstream
=======
import re

def grep(pattern, flags, files):
    result = []
    for file in files:
        with open(file) as f:
            lines = f.readlines()
            for line in lines:
                line_num = len(lines) + 1
                if flags == '-n':
                    modified_line = f"{line_num}:{line}"
                elif flags == '-l':
                    modified_line = line
                elif flags == '-i':
                    modified_line = line.lower()
                else:
                    modified_line = line
                if modified_line is not None:
                    if re.fullmatch(pattern, modified_line):
                        result.append(modified_line)
    return result
>>>>>>> Stashed changes
