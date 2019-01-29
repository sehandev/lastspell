from io import open

def check_last(option):
    with opentt('training_data/data_out.txt', 'r', encoding='UTF8') as f:
        res = ""
        for line in f:
            line = line.strip()
            last = line[-1]
            if option == 0:
                if int((ord(last) - 0xAC00) % 28) != 0:
                    line = line.split('$')[0] + '$o'
                else:
                    line = line.split('$')[0] + '$x'
            elif option == 1:
                if int((ord(last) - 0xAC00) % 28) != 0:
                    line += '$o'
                else:
                    line += '$x'
            res += line + '\n'

    res = res[:-1]  # delete blank line

    with opentt('training_data/data_out.txt', 'w', encoding='UTF8') as f:
        f.write(res)

def deduplication():
    words = []

    with open("../training_data/data_in.txt", 'r') as f:
        for line in f:
            line = line.strip().lower()
            if len(line) > 3 and line.isalpha():
                words.append(line)
            else:
                print(line)

    return(list(set(words)))

def sort():
    words = deduplication()
    words.sort()

    with open("../training_data/data_in_sorted.txt", 'w') as f:
        for word in words:
            f.write(word + '\n')

def temp():
    with open("../dict/cntlist.rev", 'r') as f:
        temp = f.readlines()
    word_arr = []
    for t in temp:
        word_arr.append(t.split('%')[0])
    with open("../training_data/data_in2.txt", 'w') as f:
        for w in word_arr:
            f.write(w+'\n')

sort()
