import java.util.ArrayList;
import java.util.List;

public class SgfParsing {
    public SgfNode parse(String input) throws SgfParsingException {
        return parseToken(input);
    }

    private List<Property> parseToken(String token) throws SgfParsingException {
        List<Property> properties = new ArrayList<>();
        String[] parts = token.split(",");
        for (String part : parts) {
            String keyPart = part.trim();
            if (!keyPart.isEmpty()) {
                String[] values = splitValues(part, '\\');
                if (values.length > 0) {
                    properties.add(new Property(keyPart, values));
                }
            }
        }
        return properties;
    }

    private String splitValues(String value, String delimiter) {
        String[] parts = value.split(delimiter);
        List<String> values = new ArrayList<>();
        for (String part : parts) {
            if (part.isEmpty()) continue;
            String valueStr = part;
            int bracketIndex = valueStr.indexOf('[');
            if (bracketIndex == -1) {
                values.add(valueStr);
            } else {
                String key = valueStr.substring(0, bracketIndex).trim();
                String value = valueStr.substring(bracketIndex + 1).trim();
                values.add(key + ":" + value);
            }
        }
        return values;
    }

    private static class Property {
        String key;
        List<String> values;

        Property(String key, List<String> values) {
            this.key = key;
            this.values = values;
        }
    }

    private static class Node {
        Property children;

        Node children() {
            return this;
        }

        public Node getRoot() {
            return this;
        }
    }

    private static Node createNode() {
        return new Node();
    }
}