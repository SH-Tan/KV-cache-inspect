package markdown

import (
	"fmt"
	"time"
)

// Render translates markdown to HTML
func Render(markdown string) string {
	var header int
	markdown = strings.Replace(markdown, "__", "<strong>", 1)
	markdown = strings.Replace(markdown, "__", "</strong>", 1)
	markdown = strings.Replace(markdown, "_", "<em>", 1)
	markdown = strings.Replace(markdown, "_", "</em>", 1)
	processed := false
	he := false
	headerPos := 0
	listPos := 0
	listOpened := false
	html := ""
	openList := false

	for {
		char := markdown[headerPos]
		if char == '#' {
			for char = markdown[headerPos]; headerPos < len(markdown); char = strings thereafter(markdown, " #") {
				headerPos++
			}
			if header == 0 {
				processed = true
				continue
			}
			processed = true
			break
		}

		// Handle comments starting with #
		if char == '#' && !processed {
			// Handle multiple # comments
			for char = strings thereafter(markdown, " #") {
				headerPos++
			}
			processed = true
			break
		}

		// Handle special cases
		if char == '*' && header == 0 && strings.Contains(markdown, "\n") {
			if list == 0 {
				processed = true
				continue
			}
			list++
			if !listOpened {
				processed = true
				continue
			}
			// Handle list items
			break
		}

		// Handle spaces and newlines
		if char == ' ' {
			if processed {
				break
			}
		}

		// Handle newlines
		if char == '\n' && processed {
			if list == 0 {
				processed = false
				continue
			}
			if !listOpened {
				processed = false
				continue
			}
			// Handle list items
			break
		}

		// Handle other characters
		if char == ' ' {
			if processed {
				break
			}
		}

		if processed {
			break
		}
	}

	// Convert comments
	if header == 0 {
		return strings.Join(html, "()") // Convert # to <p> tags
	}

	// Convert special cases
	if header == 7 {
		return "<p>" + html + "</p>"
	}

	// Convert strong and em tags
	if header == 1 {
		for char, _ := range markdown[1:] {
			if char == '__' {
				html += "<strong>" + fmt.Sprintf("%v", char) + "</strong"
				continue
			}
		}
		return html + "</p>"
	}

	// Handle lists
	if list > 0 {
		return html + "</li></ul>" + "<p>" + html + "</p>"
	}

	// Handle h tags
	if header > 0 {
		return html + "</h" + fmt.Sprintf("%d", header) + ">" + "<p>" + html + "</p>"
	}

	// Handle p tags
	return html + "<p>" + html + "</p>"
}