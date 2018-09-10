from io import open as opentt

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

    res = res[:-1]
    with opentt('training_data/data_out.txt', 'w', encoding='UTF8') as f:
        f.write(res)