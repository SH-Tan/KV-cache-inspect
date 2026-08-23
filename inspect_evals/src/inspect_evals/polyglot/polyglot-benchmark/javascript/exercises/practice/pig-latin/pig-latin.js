function translate(s) {
    let words = s.split(' ');
    let translatedWords = [];
    
    for (let word of words) {
        let translatedWord = word;
        
        if (word[0] === 'a' || word[0] === 'e' || word[0] === 'i' || word[0] === 'o' || word[0] === 'u') {
            translatedWord += 'ay';
        } else if (word[0] === 'x' || word[0] === 'y' || word[0] === 't') {
            translatedWord += 'ay';
        } else if (word[0] === 'q') {
            let consonants = word.substring(0, word.length - 2).split('').filter(c => c >= 'a' && c <= 'z' && c !== 'y').join('');
            translatedWord = consonants + word.substring(word.length - 2) + 'ay';
        } else if (word[0] === 'y') {
            let consonants = word.substring(0, word.length - 1).split('').filter(c => c >= 'a' && c <= 'z' && c !== 'y').join('');
            translatedWord = consonants + word.substring(1) + 'ay';
        } else {
            let consonants = word.substring(0, word.length - 1).split('').filter(c => c >= 'a' && c <= 'z' && c !== 'y').join('');
            translatedWord = consonants + word.substring(1) + 'ay';
        }
        
        translatedWords.push(translatedWord);
    }
    
    return translatedWords.join(' ');
}