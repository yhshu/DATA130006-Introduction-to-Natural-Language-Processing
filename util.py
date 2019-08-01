def rchop(string, substring):
    return (True, string[:-len(substring)]) if string.endswith(substring) else (False, string)


def lchop(string, substring):
    return (True, string[:-len(substring)]) if string.startswith(substring) else (False, string)


def legal_number(string):
    try:
        float(string)
    except ValueError:
        return False
    else:
        return True
