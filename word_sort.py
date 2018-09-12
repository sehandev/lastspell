def make_test(true_array, false_array, size):
    test_data = [[] for _ in range(size * 0.1)]

def make_tf_data():
    train_length = 5
    
    with open("training_data/data_out.txt", 'r') as f:
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
        if len(t) > train_length:
            t = t[:train_length]
        while len(t) < train_length:
            t += '$'
        r_last5.append(t)
    
    true_array, false_array = [], []
    for i in range(len(r_last5)):
        if syllable_check[i] == 1:
            true_array.append(r_last5[i])
        elif syllable_check[i] == 0:
            false_array.append(r_last5[i])
    
    test_data = make_test(true_array, false_array, len(r_last5))

    print("== Data ====")
    print("True : " + str(syllable_check.count(1)))
    print("False : " + str(syllable_check.count(0)))
    
    return r_last5, syllable_check, len(last5)
  
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
