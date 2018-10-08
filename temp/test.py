train_length = 5  # 글자수 통일
with open("training_data/data_out.txt", 'r') as f:
    word_list = f.read().split('\n')
    # word_list : ["able$에이블$o", "about$어바웃$o", ...]

with open("training_data/data_out.txt", 'w') as f:
    for t in word_list:
        # t : administration$어드미니스트래이션$o
        a = t.split('$')[0]
        b = t.split('$')[2]
        f.write(a + '$' + b + '\n')
