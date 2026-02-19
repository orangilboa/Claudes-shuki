def reverse_string(s, verbose=True):
    result = s[::-1]
    if verbose:
        print(f"reverse_string('{s}') -> '{result}'")
    return result


def capitalize_words(s, verbose=True):
    result = s.title()
    if verbose:
        print(f"capitalize_words('{s}') -> '{result}'")
    return result


def count_vowels(s, verbose=True):
    count = sum(1 for c in s.lower() if c in "aeiou")
    print(f"count_vowels('{s}') -> {count}")
    return count


def is_palindrome(s, verbose=True):
    cleaned = s.lower().replace(" ", "")
    result = cleaned == cleaned[::-1]
    print(f"is_palindrome('{s}') -> {result}")
    return result


def truncate(s, max_len, suffix="...", verbose=True):
    if len(s) <= max_len:
        if verbose:
            print(f"truncate('{s}', {max_len}) -> '{s}' (no truncation)")
        return s
    result = s[:max_len] + suffix
    if verbose:
        print(f"truncate('{s}', {max_len}) -> '{result}'")
    return result
