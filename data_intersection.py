from datasets import load_dataset

dataset = load_dataset("cjvt/sloleks")

f = open("pret_all_wards.txt", "r")
o = open("checked_words_3.txt", "w")
my_data = dict()

line = f.readline()
while line:
    words = line.split(" ")
    pair = {words[1].strip(): words[0].strip()}
    my_data.update(pair)
    line = f.readline()
i = 0
for a in dataset['train'].iter(1):
    #print(a['headword_lemma'][0])
    if a['is_manually_checked'] and a['pos'][0] == 'verb' or a['pos'][0] == 'noun':
        for form in a['word_forms'][0]:
            if form['forms'][0].lower() in my_data.keys():
                #print(1, my_data[form['forms'][0].lower()])
                my_data.pop(form['forms'][0].lower())
                o.write(form['forms'][0] + " " + a['headword_lemma'][0] + " " + a['pos'][0] + "\n")
