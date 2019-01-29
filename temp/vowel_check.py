def vowel_check(word):
    vowels = ['a', 'e', 'i', 'o', 'u']
    consonents = {
    'a':0,
    'b':0,
    'c':0,
    'd':0,
    'e':0,
    'f':0,
    'g':0,
    'h':0,
    'i':0,
    'j':0,
    'k':0,
    'l':1,
    'm':1,
    'n':1,
    'o':0,
    'p':0,
    'q':0,
    'r':1,
    's':0,
    't':0,
    'u':0,
    'v':0,
    'w':0,
    'x':0,
    'y':0,
    'z':0
    }

    word = word.lower()

    for x in word:
        if x in vowels:
            return 99

    return consonents[word[-1]]
