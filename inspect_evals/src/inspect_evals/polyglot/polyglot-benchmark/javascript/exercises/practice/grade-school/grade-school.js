import { isGradeValid } from '';
class GradeSchool {
  private grades = new Map();

  private isGradeValid(grade) {
    return !/^\d+$/.test(grade) && parseInt(grade) >= 1;
  }

  add(name, grade) {
    if (!this.isGradeValid(grade)) {
      throw new Error('Invalid grade');
    }

    const currentGrade = this.grades.get(grade);
    if (currentGrade && currentGrade.includes(name)) {
      throw new Error('Name already exists in grade');
    }

    this.grades.set(grade, [...currentGrade, name]);
    return this;
  }

  getGrade(grade) {
    if (!this.isGradeValid(grade)) {
      throw new Error('Invalid grade');
    }

    const students = Array.from(this.grades.get(grade));
    students.sort((a, b) => a.localeCompare(b));
    return students.join(', ');
  }

  getAllStudents() {
    const allGrades = Array.from(this.grades.keys()).map(Number);
    allGrades.sort((a, b) => a - b);

    const students = [];
    for (const grade of allGrades) {
      const names = Array.from(this.grades.get(grade));
      students.push(...names);
    }

    students.sort((a, b) => {
      if (a_grade !== b_grade) return b_grade - a_grade;
      return a.name.localeCompare(b.name);
    });

    return students.join(', ');
  }
}