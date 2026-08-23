using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

[Functional reactive]
public class Hangman {
    private Set<string> activeLetters;
    private string partiallyGuessed;
    private set<string> remainingLetters;
    private set<string> guessedLetters;

    public Observable<string> Hangman.Current() {
        return new Observable<string>() {
            get => {
                // This is a placeholder for the actual implementation
                // The correct implementation would return an Output that includes the current state
                return partiallyGuessed + " " + remainingLetters;
            }
        };
    }

    public <T> Observable<T> play(Observable<T> words, Observable<T> letters) {
        return new Observable<T>() {
            @Override
            async () {
                // Subscribe to the words and letters streams
                await subscription(words);
                await subscription(letters);

                // Process the letters
                foreach (T letter in letters.current) {
                    if (activeLetters.Contains(letter)) {
                        activeLetters.Remove(letter);
                        guessedLetters.Add(letter);
                    }
                }

                // Update the remaining letters
                remainingLetters = letters.current
                    .Where(l => !activeLetters.Contains(l))
                    .ToList();
                partialGuessed = guessedLetters.Reverse().ToArray().Concat();
            }
        }
    }

    private async Task< Observable< observable> > subscribe(Observable< observable> observer) {
        yield return () => await observer();
    }
}