import matplotlib.pyplot as plt
import nltk
nltk.download('averaged_perceptron_tagger')

with open('./training_data/data_out.txt', 'r') as f:
    lines = f.readlines()

count_length = [0 for i in range(26)]  # 단어 길이 0~25
count_last = [0, 0]  # 자음, 모음

for line in lines:
    front = line.split('$')[0]
    count_length[len(front)] += 1
    if front[-1] in "aeoiu":
        count_last[1] += 1
    else:
        count_last[0] += 1
    tagged = nltk.pos_tag(front)
    print(tagged)


print('=== length of words ======')
print(count_length)

# bar graph of length
index = range(26)
plt.subplot(1, 2, 1)
plt.bar(index, count_length)
plt.xticks(index, fontsize=7)
plt.title('length of word', fontsize=20)

print('=== Consonant & Vowel ======')
print(count_last)

# bar graph of length
index = range(2)
label = ['Consonant', 'Vowel']
plt.subplot(1, 2, 2)
plt.bar(index, count_last)
plt.xticks(index, label, fontsize=7)
plt.title('Consonant&Vowel', fontsize=20)
plt.show()
