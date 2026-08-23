export class Song {
  const versesList = [
    "I know an old lady who swallowed a fly.
    I don't know why she swallowed the fly. Perhaps she'll die.",
    "I know an old lady who swallowed a spider.
    It wriggled and jiggled and tickled inside her.
    She swallowed the spider to catch the fly.
    I don't know why she swallowed the fly. Perhaps she'll die.",
    "I know an old lady who swallowed a bird.
    How absurd to swallow a bird! She swallowed the bird to catch the spider that wriggled and jiggled and tickled inside her.
    She swallowed the spider to catch the fly.
    I don't know why she swallowed the fly. Perhaps she'll die.",
    "I know an old lady who swallowed a cat.
    Imagine that, to swallow a cat! She swallowed the cat to catch the bird.
    She swallowed the bird to catch the spider that wriggled and jiggled and tickled inside her.
    She swallowed the spider to catch the fly.
    I don't know why she swallowed the fly. Perhaps she'll die.",
    "I know an old lady who swallowed a dog.
    What a hog, to swallow a dog! She swallowed the dog to catch the cat.
    She swallowed the cat to catch the bird.
    She swallowed the bird to catch the spider that wriggled and jiggled and tickled inside her.
    She swallowed the spider to catch the fly."
  ];

  // Function to return the first verse
  verse() {
    return versesList[0];
  }

  // Function to return all verses
  verses() {
    return versesList.join('\n');
  }
}