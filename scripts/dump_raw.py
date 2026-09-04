import sys
data = open(sys.argv[1], "rb").read().decode("utf-8", "replace")
print(repr(data))
