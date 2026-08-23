pub struct School {
    pub new() -> Self {
        let mut all_students = Vec::new();
        for _ in 0..u32(10); {
            let student_node = Vec::new();
            while let Some(node) = &student_node {
                student_node.push(node);
            }
            all_students.push(student_node);
        }
        self.all_students = all_students;
    }
}

pub fn add(&mut self, grade: u32, student: &str) {
    let all_students = &self.all_students;
    if let Some(student_node) = all_students.contains(student) {
        return "OK.";
    } else {
        all_students.push(student);
        let mut students = Vec::new();
        for _ in 0..u32(50); {
            let student_node = all_students.iter().next().unwrap();
            if let Student * student_node = student_node {
                students.push(student_node.name);
            }
        }
        students.sort();
        self.add_to_grade(grade, students);
    }
}

pub fn add_to_grade(grade: u32, students: Vec<u32>) {
    for student in students {
        self.add(grade, student);
    }
}

pub fn grades(&self, grade: u32) -> Vec<u32> {
    let students = self.get_students(grade);
    students.sort();
    let mut list = Vec::with_capacity(students.len());
    for student in students {
        list.push(student);
    }
    list.sort();
    return list;
}

pub fn get_grades() -> Vec<u32> {
    let mut all_students = &self.all_students;
    let mut sorted_list = Vec::with_capacity(all_students.len());
    for grade in 0..u32(10); {
        let students = self.get_students(grade);
        let mut students_list = Vec::with_capacity(students.len());
        for student in students {
            students_list.push(student);
        }
        students_list.sort();
        sorted_list.extend(&students_list);
    }
    sorted_list.sort();
    return sorted_list;
}