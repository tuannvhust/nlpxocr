import os 
with open('data/test_output_gt_sentences_v2.txt','r') as file:
    train_set = [line.strip('\n') for line in file]
    open('data/vi.test.txt','w')
with open('data/vi.test.txt','a') as write_file:
        j=1
        k=2
        while j < len(train_set) and k< len(train_set):
            new_line = train_set[j].replace('|',' ').replace('"','')+"|"+train_set[k].replace('|',' ').replace('"','')
            write_file.writelines(new_line+'\n')
            j=j+3
            k=k+3