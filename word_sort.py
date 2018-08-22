def new_words():
    with open("training_data/words_data.txt", 'r') as f:
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
            
def make_tf_data():
    with open("training_data/words_data.txt", 'r') as f:
        word_list = f.read().split('\n')
        # word_list : ["able$에이블$o", "about$어바웃$o", ...]
    
    last5 = []
    syllable_check = []
    for t in word_list:
        # t : administration$어드미니스트래이션$o
        last5.append(t.split('$')[0])
        
        cy = t.split('$')[2]
        if cy == 'o':
            syllable_check.append(1)
        elif cy == 'x':
            syllable_check.append(0)
            
    # last5 : ["able", "about", ...]
    # syllable_check : [1, 1, ...]
    r_last5 = []
    for t in last5:
        # t : "able"
        t = t[::-1]
        if len(t) > 5:
            t = t[:5]
        while len(t) < 5:
            t += '$'
        r_last5.append(t)
    
#     print('last5 length: ' + str(len(r_last5)))
#     print('syllable_check length: ' + str(len(syllable_check)))
    return r_last5, syllable_check, len(last5)

def deduplication():
    with open("training_data/data_in.txt", 'r') as f:
        words = []
        for line in f:
            words.append(line.strip().lower())
            
        return(list(set(words)))
        
def sort():
    words = deduplication()
    words.sort()
    
    with open("training_data/data_in_sorted.txt", 'w') as f:
        for word in words:
            f.write(word + '\n')
            
if __name__ == "__main__":
    sort()
