from string_utils import reverse_string, capitalize_words, count_vowels, is_palindrome, truncate
from math_utils import clamp, factorial, is_prime, average, percentage
from list_utils import flatten, deduplicate, chunk, rotate, group_by

def demo_string_utils(verbose=True):
    if verbose:
        print("=== string_utils ===")
    reverse_string("hello world", verbose=verbose)
    capitalize_words("the quick brown fox", verbose=verbose)
    count_vowels("hello world", verbose=verbose)
    is_palindrome("racecar", verbose=verbose)
    is_palindrome("hello", verbose=verbose)
    truncate("this is a long sentence", 10, verbose=verbose)
    print()


def demo_math_utils(verbose=True):
    print("=== math_utils ===")
    clamp(15, 0, 10, verbose=verbose)
    factorial(6, verbose=verbose)
    is_prime(17, verbose=verbose)
    is_prime(18, verbose=verbose)
    average([4, 8, 15, 16, 23, 42], verbose=verbose)
    percentage(45, 200, verbose=verbose)
    print()


def demo_list_utils(verbose=True):
    if verbose:
        print("=== list_utils ===")
    flatten([1, [2, 3], [4, [5, 6]]], verbose=verbose)
    deduplicate([1, 2, 2, 3, 1, 4], verbose=verbose)
    chunk([1, 2, 3, 4, 5, 6, 7], 3, verbose=verbose)
    rotate([1, 2, 3, 4, 5], 2, verbose=verbose)
    group_by([1, 2, 3, 4, 5, 6], lambda x: "even" if x % 2 == 0 else "odd", verbose=verbose)
    print()


if __name__ == "__main__":
    demo_string_utils()
    demo_math_utils()
    demo_list_utils()
