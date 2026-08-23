function say(n) {
  if (n < 0 || n >= 1000000000000) {
    throw new Error('Invalid number');
  }

  function convertThreeDigits(num) {
    if (num === 0) return 'zero';
    const lessThanTwenty = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen'];
    const twentyIndex = num < 20 ? num : lessThanTwenty[num];
    
    let result = '';
    if (num >= 20) {
      result += lessThanTwenty[num - 20];
      result += ' ' + convertLessThanTwenty(num % 20);
    } else {
      result = lessThanTwenty[num];
    }
    return result;
  }

  function convertLessThanTwenty(num) {
    const lessThanTwenty = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen'];
    return lessThanTwenty[num];
  }

  function splitIntoChunks(num) {
    const numStr = num.toString().padStart(12, '0');
    const chunks = [];
    for (let i = 0; i < numStr.length; i += 3) {
      const start = i;
      const end = start + 3;
      const chunk = numStr.substring(start, end);
      if (chunk === '') chunk = '0';
      chunks.push(parseInt(chunk, 10));
    }
    return chunks;
  }

  const chunks = splitIntoChunks(n);
  const processedChunks = [];
  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];
    if (chunk === 0) continue;

    const words = convertThreeDigits(chunk);
    if (words) {
      let scale;
      if (i === 0) scale = '';
      else if (i === 1) scale = ' thousand';
      else if (i === 2) scale = ' million';
      else if (i === 3) scale = ' billion';
      processedChunks.push(`${words} ${scale}`);
    }
  }

  return processedChunks.join(' ');
}