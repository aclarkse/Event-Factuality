with open('fb_factValue.txt') as f:
    with open('factValue.txt', 'w') as f1:
        for line in f:
            tokens = line.split("|||")
            source = tokens[0]
            sentence_ID = tokens[1]
            event = tokens[6]
            fact_annot = tokens[8]
            f1.write(source + ', ' + sentence_ID + ', ' + event + ', ' + fact_annot)
f1.close()
f.close()
