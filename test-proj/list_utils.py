def flatten(nested, verbose=True):
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten(item, verbose=verbose))
        else:
            result.append(item)
    if verbose:
        print(f"flatten({nested}) -> {result}")
    return result


def deduplicate(items, verbose=True):
    seen = set()
    result = [x for x in items if not (x in seen or seen.add(x))]
    if verbose:
        print(f"deduplicate({items}) -> {result}")
    return result


def chunk(items, size, verbose=True):
    result = [items[i:i + size] for i in range(0, len(items), size)]
    if verbose:
        print(f"chunk({items}, {size}) -> {result}")
    return result


def rotate(items, steps, verbose=True):
    if not items:
        if verbose:
            print(f"rotate([], {steps}) -> []")
        return []
    n = len(items)
    steps = steps % n
    result = items[steps:] + items[:steps]
    if verbose:
        print(f"rotate({items}, {steps}) -> {result}")
    return result


def group_by(items, key_fn, verbose=True):
    groups = {}
    for item in items:
        k = key_fn(item)
        groups.setdefault(k, []).append(item)
    if verbose:
        print(f"group_by({items}) -> {groups}")
    return groups
