import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class PhoneNumber {
    public String getNumber() {
        // Remove all non-digit characters from the input string
        String numStr = numberString.replaceAll("[^0-9]", "");
        
        // Ensure the string is at least 10 characters long to form a valid NANP number
        if (numStr.length() < 10) {
            return numStr; // Not enough digits, return as is
        }
        
        // Split into area code (first three), exchange code (next three), and subscriber (last four)
        String area = numStr.substring(0, 3);
        String exchange = numStr.substring(3, 6);
        String subscriber = numStr.substring(6);
        
        // Concatenate them to form the final NANP number
        return area + exchange + subscriber;
    }
}