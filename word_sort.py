import numpy as np
import random

np.set_printoptions(threshold=np.inf)

def lastspell_data():
    train_length = 5
    with open("training_data/data_out.txt", 'r') as f:
        word_list = f.read().split('\n')
        # word_list : ["able$에이블$o", "about$어바웃$o", ...]
    
    full_length = len(word_list)  # 단어 개수

    pre_true_array = []
    pre_false_array = []

    for t in word_list:
        # t : administration$어드미니스트래이션$o
        ox = t.split('$')[2]
        if ox == 'o':
            pre_true_array.append(t.split('$')[0])
        elif ox == 'x': 
            pre_false_array.append(t.split('$')[0])

    # precess data / ex) "information" > "noita"
    true_array = processng_data(pre_true_array, train_length)
    false_array = processng_data(pre_false_array, train_length)

    #make test data
    test_data = []
    for i in range(int(full_length * 0.05)):
        test_data.append([true_array.pop(), 1])
        test_data.append([false_array.pop(), 0])

    # make train data
    train_data = []
    for i in true_array:
        train_data.append([i, 1])
    for i in false_array:
        train_data.append([i, 0])

    # shuffle datas
#    random.shuffle(train_data)
#    random.shuffle(test_data)
 
    # sperate input data and target data
    train_input = []
    train_target = []
    for i in test_data:
        train_input.append(i[0])
        train_target.append(i[1])

    test_input = []
    test_target = []
    for i in test_data:
        test_input.append(i[0])
        test_target.append(i[1])

    # change array to eye array
    train_input = data_to_eye(train_input)
    test_input = data_to_eye(test_input)

    return [train_input, train_target], [test_input, test_target]


def processng_data(arr, train_length):
    # arr : ["able", "about", ... ]
    result_arr = []
    for t in arr:
        # t : "able"
        t = t[::-1]
        if len(t) > train_length:
            t = t[:train_length]
        while len(t) < train_length:
            t += '$'
        result_arr.append(t)
    return result_arr
 

def data_to_eye(arr):
    eng = ['$', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    dic_eng = {n: i for i, n in enumerate(eng)}
    # dic_eng : {'$' : 0, 'a' : 1, 'b' : 2, 'c' : 3, ..., 'j' : 10, 'k' : 11, ...}
    print(dic_eng)

    temp = []
    for t in arr:
        input = [dic_eng[i] for i in t]
        temp.append(np.eye(27)[input])
    return temp


def deduplication():
    words = []
    
    with open("training_data/data_in.txt", 'r') as f:
        for line in f:
            line = line.strip().lower()
            if len(line) > 2 and line.isalpha():
                words.append(line)
            else:
                print(line)
                
    return(list(set(words)))
        
def sort():
    words = deduplication()
    words.sort()
    
    with open("training_data/data_in_sorted.txt", 'w') as f:
        for word in words:
            f.write(word + '\n')
            
if __name__ == "__main__":
    sort()
