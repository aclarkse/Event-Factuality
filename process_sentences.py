with open('fb_sentences.txt') as f:
    with open('sentences.txt', 'w') as f1:
        for line in f:
            tokens = line.split("|||")
            source = tokens[0]
            sentence_ID = tokens[1]
            sentence = tokens[2]
            f1.write(source + '|' + sentence_ID + '|' + sentence)
f1.close()
f.close()
