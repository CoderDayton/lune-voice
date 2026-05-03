//! Prompt output cleanup utilities.

/// Collapses unnecessary spaces within numeric values like thousands separators.
///
/// Handles:
/// - Comma spacing: `2, 000` → `2,000`
/// - European thousand sep: `2 000` → `2000` (when 3+ digit groups)
/// - Decimal periods: `3 . 14` → `3.14`
///
/// Spaces are preserved elsewhere in the text (e.g. `2 apples` remains unchanged).
pub fn fix_all_number_spaces(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        // Pattern 0: digit + space + comma + [spaces] + digit
        if i > 0
            && i + 2 < chars.len()
            && chars[i - 1].is_ascii_digit()
            && chars[i] == ' '
            && chars[i + 1] == ','
        {
            let j = skip_spaces(&chars, i + 2);
            if j < chars.len() && chars[j].is_ascii_digit() {
                result.push(',');
                i = j;
                continue;
            }
        }

        // Pattern 0b: digit + space + period + [spaces] + digit
        if i > 0
            && i + 2 < chars.len()
            && chars[i - 1].is_ascii_digit()
            && chars[i] == ' '
            && chars[i + 1] == '.'
        {
            let j = skip_spaces(&chars, i + 2);
            if j < chars.len() && chars[j].is_ascii_digit() {
                result.push('.');
                i = j;
                continue;
            }
        }

        // Pattern 1: digit + comma + [spaces] + digit -> collapse spaces
        if i + 1 < chars.len() && chars[i].is_ascii_digit() && chars[i + 1] == ',' {
            let j = skip_spaces(&chars, i + 2);
            if j < chars.len() && chars[j].is_ascii_digit() {
                result.push(chars[i]);
                result.push(',');
                i = j;
                continue;
            }
        }

        // Pattern 2: digit + comma + space[s] + digit
        if i + 2 < chars.len()
            && i > 0
            && chars[i - 1].is_ascii_digit()
            && chars[i] == ','
            && chars[i + 1] == ' '
            && chars[i + 2].is_ascii_digit()
        {
            result.push(',');
            i = skip_spaces(&chars, i + 2);
            continue;
        }

        // Pattern 3: digit + period + [spaces] + digit -> collapse spaces (decimals)
        if i + 1 < chars.len() && chars[i].is_ascii_digit() && chars[i + 1] == '.' {
            let j = skip_spaces(&chars, i + 2);
            if j < chars.len() && chars[j].is_ascii_digit() {
                result.push(chars[i]);
                result.push('.');
                i = j;
                continue;
            }
        }

        // Pattern 4: digit + period + space[s] + digit
        if i + 2 < chars.len()
            && i > 0
            && chars[i - 1].is_ascii_digit()
            && chars[i] == '.'
            && chars[i + 1] == ' '
            && chars[i + 2].is_ascii_digit()
        {
            result.push('.');
            i = skip_spaces(&chars, i + 2);
            continue;
        }

        // Pattern 5: European thousand separator (digit + spaces + 3 digits)
        if i + 1 < chars.len() && chars[i].is_ascii_digit() {
            let j = skip_spaces(&chars, i + 1);
            if j + 2 < chars.len()
                && chars[j].is_ascii_digit()
                && chars[j + 1].is_ascii_digit()
                && chars[j + 2].is_ascii_digit()
                && (j - i - 1) > 0
            // there were spaces
            {
                result.push(chars[i]);
                i = j;
                continue;
            }
        }

        result.push(chars[i]);
        i += 1;
    }
    result
}

/// Skip whitespace characters, returning the index of the first non-space char.
fn skip_spaces(chars: &[char], start: usize) -> usize {
    let mut j = start;
    while j < chars.len() && chars[j] == ' ' {
        j += 1;
    }
    j
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fix_all_number_spaces_comma() {
        assert_eq!(fix_all_number_spaces("2, 000"), "2,000");
        assert_eq!(fix_all_number_spaces("2 , 100"), "2,100");
    }

    #[test]
    fn fix_all_number_spaces_decimal() {
        assert_eq!(fix_all_number_spaces("3 . 14"), "3.14");
        assert_eq!(fix_all_number_spaces("pi is 3 . 14159"), "pi is 3.14159");
    }

    #[test]
    fn fix_all_number_spaces_european() {
        // European: 2 000 = 2000
        assert_eq!(fix_all_number_spaces("2 000 000"), "2000000");
    }

    #[test]
    fn fix_all_number_spaces_preserves_text() {
        assert_eq!(fix_all_number_spaces("2 apples"), "2 apples");
        assert_eq!(fix_all_number_spaces("hello, 2 apples"), "hello, 2 apples");
        assert_eq!(
            fix_all_number_spaces("hello , 2 apples"),
            "hello , 2 apples"
        );
    }
}
