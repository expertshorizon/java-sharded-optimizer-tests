package com.trainingsim.json;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class Json {

    private final String text;
    private int pos;

    private Json(String text) {
        this.text = text;
        this.pos = 0;
    }

    public static Object parse(String text) {
        Json parser = new Json(text);
        parser.skipWhitespace();
        Object value = parser.parseValue();
        parser.skipWhitespace();
        if (parser.pos != parser.text.length()) {
            throw new JsonException("trailing content at position " + parser.pos);
        }
        return value;
    }

    @SuppressWarnings("unchecked")
    public static Map<String, Object> parseObject(String text) {
        return (Map<String, Object>) parse(text);
    }

    private Object parseValue() {
        skipWhitespace();
        if (pos >= text.length()) {
            throw new JsonException("unexpected end of input");
        }
        char c = text.charAt(pos);
        switch (c) {
            case '{':
                return parseObjectValue();
            case '[':
                return parseArrayValue();
            case '"':
                return parseStringValue();
            case 't':
                expectLiteral("true");
                return Boolean.TRUE;
            case 'f':
                expectLiteral("false");
                return Boolean.FALSE;
            case 'n':
                expectLiteral("null");
                return null;
            default:
                return parseNumberValue();
        }
    }

    private Map<String, Object> parseObjectValue() {
        Map<String, Object> result = new LinkedHashMap<>();
        pos++;
        skipWhitespace();
        if (peek() == '}') {
            pos++;
            return result;
        }
        while (true) {
            skipWhitespace();
            String key = parseStringValue();
            skipWhitespace();
            expectChar(':');
            Object value = parseValue();
            result.put(key, value);
            skipWhitespace();
            char next = peek();
            if (next == ',') {
                pos++;
                continue;
            }
            if (next == '}') {
                pos++;
                break;
            }
            throw new JsonException("expected ',' or '}' at position " + pos);
        }
        return result;
    }

    private List<Object> parseArrayValue() {
        List<Object> result = new ArrayList<>();
        pos++;
        skipWhitespace();
        if (peek() == ']') {
            pos++;
            return result;
        }
        while (true) {
            Object value = parseValue();
            result.add(value);
            skipWhitespace();
            char next = peek();
            if (next == ',') {
                pos++;
                continue;
            }
            if (next == ']') {
                pos++;
                break;
            }
            throw new JsonException("expected ',' or ']' at position " + pos);
        }
        return result;
    }

    private String parseStringValue() {
        expectChar('"');
        StringBuilder sb = new StringBuilder();
        while (true) {
            if (pos >= text.length()) {
                throw new JsonException("unterminated string");
            }
            char c = text.charAt(pos++);
            if (c == '"') {
                break;
            }
            if (c == '\\') {
                char escaped = text.charAt(pos++);
                switch (escaped) {
                    case '"':
                        sb.append('"');
                        break;
                    case '\\':
                        sb.append('\\');
                        break;
                    case '/':
                        sb.append('/');
                        break;
                    case 'n':
                        sb.append('\n');
                        break;
                    case 't':
                        sb.append('\t');
                        break;
                    case 'r':
                        sb.append('\r');
                        break;
                    case 'b':
                        sb.append('\b');
                        break;
                    case 'f':
                        sb.append('\f');
                        break;
                    case 'u':
                        String hex = text.substring(pos, pos + 4);
                        sb.append((char) Integer.parseInt(hex, 16));
                        pos += 4;
                        break;
                    default:
                        throw new JsonException("invalid escape at position " + pos);
                }
            } else if (c < 0x20) {
                // JSON forbids unescaped control characters inside a string.
                // Accepting them would let a malformed log turn into a valid
                // one: a literal newline inside "1\n" survives parsing and
                // then trims away, so a document that should be rejected
                // decodes as the finite gradient 1 instead.
                throw new JsonException(
                        "unescaped control character U+" + String.format("%04X", (int) c)
                                + " in string at position " + (pos - 1));
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    private Long parseNumberValue() {
        int start = pos;
        if (peek() == '-') {
            pos++;
        }
        int digitsStart = pos;
        while (pos < text.length() && text.charAt(pos) >= '0' && text.charAt(pos) <= '9') {
            pos++;
        }
        int digits = pos - digitsStart;
        if (digits == 0) {
            throw new JsonException("expected a digit at position " + digitsStart);
        }
        // JSON forbids a leading zero on a multi-digit integer, so "01" is not a
        // number. Accepting it would let a malformed document decode as a valid
        // one (worker id 01 silently becoming 1).
        if (digits > 1 && text.charAt(digitsStart) == '0') {
            throw new JsonException("leading zero in number at position " + digitsStart);
        }
        return Long.parseLong(text.substring(start, pos));
    }

    private void expectLiteral(String literal) {
        if (!text.startsWith(literal, pos)) {
            throw new JsonException("expected '" + literal + "' at position " + pos);
        }
        pos += literal.length();
    }

    private void expectChar(char expected) {
        skipWhitespace();
        if (pos >= text.length() || text.charAt(pos) != expected) {
            throw new JsonException("expected '" + expected + "' at position " + pos);
        }
        pos++;
    }

    private char peek() {
        skipWhitespace();
        if (pos >= text.length()) {
            throw new JsonException("unexpected end of input");
        }
        return text.charAt(pos);
    }

    private void skipWhitespace() {
        // JSON permits exactly these four characters between tokens.
        // Character.isWhitespace would also accept form feed, vertical tab and
        // a range of Unicode separators, none of which are legal JSON.
        while (pos < text.length()) {
            final char c = text.charAt(pos);
            if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
                break;
            }
            pos++;
        }
    }

    public static String write(Object value) {
        StringBuilder sb = new StringBuilder();
        writeValue(value, sb);
        return sb.toString();
    }

    @SuppressWarnings("unchecked")
    private static void writeValue(Object value, StringBuilder sb) {
        if (value == null) {
            sb.append("null");
        } else if (value instanceof Map) {
            writeObject((Map<String, Object>) value, sb);
        } else if (value instanceof List) {
            writeArray((List<Object>) value, sb);
        } else if (value instanceof String) {
            writeString((String) value, sb);
        } else if (value instanceof Boolean) {
            sb.append(value.toString());
        } else if (value instanceof Number) {
            sb.append(value.toString());
        } else {
            throw new JsonException("cannot serialize value of type " + value.getClass());
        }
    }

    private static void writeObject(Map<String, Object> map, StringBuilder sb) {
        sb.append('{');
        boolean first = true;
        for (Map.Entry<String, Object> entry : map.entrySet()) {
            if (!first) {
                sb.append(',');
            }
            first = false;
            writeString(entry.getKey(), sb);
            sb.append(':');
            writeValue(entry.getValue(), sb);
        }
        sb.append('}');
    }

    private static void writeArray(List<Object> list, StringBuilder sb) {
        sb.append('[');
        boolean first = true;
        for (Object item : list) {
            if (!first) {
                sb.append(',');
            }
            first = false;
            writeValue(item, sb);
        }
        sb.append(']');
    }

    private static void writeString(String value, StringBuilder sb) {
        sb.append('"');
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            switch (c) {
                case '"':
                    sb.append("\\\"");
                    break;
                case '\\':
                    sb.append("\\\\");
                    break;
                case '\n':
                    sb.append("\\n");
                    break;
                case '\r':
                    sb.append("\\r");
                    break;
                case '\t':
                    sb.append("\\t");
                    break;
                default:
                    sb.append(c);
            }
        }
        sb.append('"');
    }
}
