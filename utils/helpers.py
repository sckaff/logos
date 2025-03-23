def format_float(v):
    """ Format float to 4 decimal places, or 3 if ends with 0. """
    s = '{:.4f}'.format(v)
    if s[-1] == '0':
        s = s[:-1]
    return s