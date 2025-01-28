codes = ("BE", "DE", "FI", "GB", "IT", "LU", "NL", "PL")
names = ("Belgium", "Germany", "Finland", "Great Britain", "Italy", "Luxembourg", "The Netherlands", "Poland")

def name(code):
    return names[codes.index(code.upper())]
