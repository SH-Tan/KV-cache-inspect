<<<<<<< Updated upstream
=======
class School:
    def __init__(self):
        self.grades = {}  # Key: grade, Value: set of student names
        self.added = set()  # Set of added student names

    def add_student(self, name, grade):
        if name in self.added:
            raise ValueError("Name already exists")
        self.added.add(name)
        self.grades[grade].add(name)
        return "OK."

    def roster(self):
        students = []
        for g in self.grades:
            students.extend(self.grades[g])
        students.sort()
        sorted_grades = sorted(self.grades.keys())
        grouped = {}
        for g in sorted_grades:
            students_in_g = self.grades[g]
            students_in_g_sorted = sorted(students_in_g)
            grouped[g] = students_in_g_sorted
        result = []
        for g in sorted_grades:
            result.append(','.join(students_in_g_sorted))
        return ','.join(result)

    def grade(self, grade):
        if grade not in self.grades:
            raise ValueError("Grade not found")
        students = sorted(self.grades[grade])
        return "Who has the following students in grade {}? {}".format(grade, ', '.join(students))
>>>>>>> Stashed changes
