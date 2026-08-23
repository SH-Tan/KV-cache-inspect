function parallelLetterFrequency(texts) {
  const counts = {};

  // Split each text into an array of characters
  const textArrays = texts.map(text => text.split(''));

  // Now, process each character in parallel across all texts
  const allChars = textArrays.flat();

  // Use the worker thread to process all characters
  const worker = new ( () => {
    for (const char of allChars) {
      counts[char] = (counts[char] || 0) + 1;
    }
  }());

  // Wait for the worker to process all characters
  const result = await worker();

  return counts;
}