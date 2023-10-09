f = open("out_checked_words_2.txt", "r")

lines = f.readlines()
all = 0
verbs = 0
for line in lines:
    line = line.split(" ")
    lemma, word, pos = line[0].lower(), line[1].lower(), line[2].strip()
    if pos == 'verb':
        verbs += 1
    all += 1

print("verbs: ", verbs/all * 100)
print("nouns: ", (1 - verbs/all) * 100)

# verb == 26% nouns == 74% <-- checked words 3
# verb == 27% nouns == 73% <-- out checked words 3


