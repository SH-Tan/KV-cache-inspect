import "polyglot-benchmark"

func Song() string {
    var verses = []string{
        "99 bottles of beer on the wall, 99 bottles of beer.",
        "Take one down and pass it around, 98 bottles of beer on the wall.",
        "98 bottles of beer on the wall, 98 bottles of beer.",
        "Take one down and pass it around, 97 bottles of beer on the wall.",
        // ... and so on for each verse ...
    }

    var current = 99
    var nextCount = 98

    for ; current > 0; nextCount-- {
        var verse string
        if current != 1 {
            verse = "Take one down and pass it around, " + verses[current-1]
        } else {
            verse = "Take one down and pass it around, " + verses[0]
        }
        verse += " " + fmt.Sprintf("%d bottles of beer on the wall", current-1)
        verse += "."

        if current != 1 {
            verse += " and take one down and pass it around, " + verses[current-2]
        } else {
            verse += " and take one down and pass it around, " + verses[0]
        }
        verse += " " + fmt.Sprintf("%d bottles of beer on the wall", current-2)
        verse += "."

        if current != 1 {
            verse += " and take one down and pass it around, " + verses[current-3]
        } else {
            verse += " and take one down and pass it around, " + verses[0]
        }
        verse += " " + fmt.Sprintf("%d bottles of beer on the wall", current-3)
        verse += "."

        if current != 1 {
            verse += " and take one down and pass it around, " + verses[current-4]
        } else {
            verse += " and take one down and pass it around, " + verses[0]
        }
        verse += " " + fmt.Sprintf("%d bottles of beer on the wall", current-4)
        verse += "."

        verse += " and take one down and pass it around, " + verses[current-5]
        verse += " " + fmt.Sprintf("%d bottles of beer on the wall", current-5)
    }

    // Handle the case when there's only one bottle left
    if current == 1 {
        verse = "Take one down and pass it around, 0 bottles of beer on the wall."
        verse += " and take one down and pass it around, no more bottles of beer on the wall."
        verse += "Go to the store and buy more, 99 bottles of beer on the wall."
    } else {
        verse = "Take one down and pass it around, " + verses[current-1]
        verse += " " + fmt.Sprintf("%d bottles of beer on the wall", current-1)
        verse += "."
    }

    if current == 1 {
        verse += " and take one down and pass it around, 0 bottles of beer on the wall."
        verse += " and take one down and pass it around, no more bottles of beer on the wall."
        verse += "Go to the store and buy more, 99 bottles of beer on the wall."
    }

    // Final verse when there are no more bottles
    verse = "No more bottles of beer on the wall, no more bottles of beer."
    verse += "Go to the store and buy more, 99 bottles of beer on the wall."

    // Add the final verse
    verses = append(verses, verse)

    // Add the final verse
    verses = append(verses, "No more bottles of beer on the wall, no more bottles of beer.")
    verses = append(verses, "Go to the store and buy more, 99 bottles of beer on the wall.")

    return verses
}

func verse(n int) (string, error) {
    if n < 1 {
        return "No bottles of beer on the wall, no bottles of beer. Go to the store and buy more, 99 bottles of beer on the wall.", nil
    }

    if n == 1 {
        return "1 bottle of beer on the wall, 1 bottle of beer. Take one down and pass it around, 0 bottles of beer on the wall. Go to the store and buy more, 99 bottles of beer on the wall.", nil
    }

    current := n - 1
    if current == 1 {
        return "1 bottle of beer on the wall, 1 bottle of beer. Take one down and pass it around, 0 bottles of beer on the wall. Go to the store and buy more, 99 bottles of beer on the wall.", nil
    }

    return "Take one down and pass it around, " + verse(n-1)
}

func Verses(start, stop int) (string, error) {
    if start > stop {
        return "Invalid range, start must be <= stop", nil
    }

    verses = []string{
        "99 bottles of beer on the wall, 99 bottles of beer.",
        "Take one down and pass it around, " + verse(98),
    }

    for i in start+1 ...stop {
        verses = append(verses, "Take one down and pass it around, " + verse(i)),
    }

    if start == stop {
        verses = append(verses, "No more bottles of beer on the wall, no more bottles of beer.")
        verses = append(verses, "Go to the store and buy more, 99 bottles of beer on the wall.")
    }

    return verses
}

package polyglot-benchmark

func main() {
    var result = polyglot.benchmark.Run("99 Bottles of Beer on the Wall", Song)
    fmt.Printf("Time: %lg seconds\n", result.Time())
}