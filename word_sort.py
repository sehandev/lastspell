import numpy as np
import random

def lastspell_data():
    '''
    < Function >
    Make train, test data with words data file.
    -
    Make test data with same rate of true, false.
    Shuffle data.

    < Return >
    [train_input, train_target]
    train_input : array of train words.
    train_target : array of train answers.
    [test_input, test_target]
    test_input : array of test words.
    test_target : array of test answers.
    '''
    train_length = 5  # 글자수 통일
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
    random.shuffle(train_data)
    random.shuffle(test_data)

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
    '''
    < Function >
    Change words to datas of same length.

    < Parameter >
    arr : array of words. ex) ["able", "about", ... ]
    train_length : int data of limit length. Every words will be sliced to the length.

    < Return >
    result_arr : array of words of same length. 
    '''
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
    '''
    < Function >
    Change words to numpy's eye arrays.

    < Parameter >
    arr : array of words. Words are composed with small alphabet and '$'(=blank).

    < Return >
    temp : array of numpy's eye arrays.
    '''
    eng = ['$', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    dic_eng = {n: i for i, n in enumerate(eng)}
    # dic_eng : {'$' : 0, 'a' : 1, 'b' : 2, 'c' : 3, ..., 'j' : 10, 'k' : 11, ...}
    print(dic_eng)

    temp = []
    for t in arr:
        input = [dic_eng[i] for i in t]
        temp.append(np.eye(27)[input])
    return temp

