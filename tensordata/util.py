def split(str, sep, pos):
    str = str.split(sep)
    return sep.join(str[:pos]), sep.join(str[pos:])