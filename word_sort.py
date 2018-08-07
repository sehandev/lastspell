 with open("words_data.txt", 'r') as f:
    word_list = f.read().lower().split('\n')
    word_list.sort()

check_word = []
remove_word = []
for i in word_list:
    t_check = i.split('$')[0]
    # 이미 있는 단어라면 remove_word에 추가
    if check_word.count(t_check) > 0:
        remove_word.append(i)
    # 없는 단어라면 check_word에 추가
    else:
        check_word.append(t_check)

for i in remove_word:
    word_list.remove(i)

with open("words_data.txt", 'w') as f:
    for t in word_list:
        f.write(t+'\n')