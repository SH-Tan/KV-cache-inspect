public struct Robot {
    pub fn new() -> Self {
        if robot_names.len() == 0 {
            let count = robot_names.len() + 1;
            let name = format!("%s%03d", String::from("A".repeat(2)), count);
            robot_names.push(name);
            self.name = name;
        }
    }

    pub fn reset_name(&mut self, new_name) -> String {
        if let Some(name) = &self.name {
            if robot_names.iter().find(|n| n == new_name) {
                let count = robot_names.len() + 1;
                let new_name = format!("%s%03d", String::from("A".repeat(2)), count);
                self.name = String::from(new_name);
            } else {
                self.name = String::from(new_name);
            }
        }
    }
}

static let mut robot_names: [u8; 256] = [];