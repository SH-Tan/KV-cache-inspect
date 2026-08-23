class School:
    def __init__(self):
        self.students = set()

    def add_student(self, name, grade):
        if name in self.students:
            return f"Add {name} to grade {grade}. OK."
        else:
            self.students.add(name)
            return f"Add {name} to grade {grade}. OK."

    def roster(self):
        sorted_students = sorted(self.students, key=lambda x: (x['grade'], x['name']))
        return ", ".join([s['name'] for s in sorted_students])

    def grade(self, grade_number):
        students = [s for s in self.students if s['grade'] == grade_number]
        students_sorted = sorted(students, key=lambda x: x['name'])
        return ", ".join([s['name'] for s in students_sorted])

    def added(self, name):
        return name in self.students