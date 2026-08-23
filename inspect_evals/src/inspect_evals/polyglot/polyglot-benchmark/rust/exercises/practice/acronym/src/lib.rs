pub fn abbreviate(phrase: &str) -> String {
    let words = phrase.split().map(|s| s.trim()).collect();
    let mut result = Vec::new();
    
    for word in words {
        let processed_word = word
            .chars()
            .filter(|c| c.isalpha())
            .collect()
            .join()
            .as_str()
            .take(3);
        result.push(processed_word);
    }
    
    let result_str = result.join('');
    result_str
}