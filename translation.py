def checkTrait(c):
    return (int((ord(c) - 0xAC00) % 28) != 0)


print(checkTrait("칼"))