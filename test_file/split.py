import random



with open('test.txt','r',encoding='utf-8') as f:
    data = [line.strip() for line in f if random.random() < 0.4]

with open('example.txt','w',encoding='utf-8') as f:
    f.write('\n'.join(data))
