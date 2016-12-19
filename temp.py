import os

f = open('train_data.txt', 'a')
c = open('cval_data.txt', 'a')
t = open('test_data.txt' , 'a')

for (path, dirnames, filenames) in os.walk('data/angry'):
    i = 0.6 * len(filenames)
    for name in filenames[:int(i)]:
        f.write(os.path.join(path, name) + ' 0\n')
    other = filenames[int(i):]
    i = 0.5 * len(other)
    for name in other[:int(i)]:
        c.write(os.path.join(path, name) + ' 0\n')
    for name in other[int(i):]:
        t.write(os.path.join(path, name) + ' 0\n')
for (path, dirnames, filenames) in os.walk('data/disgust'):
    i = 0.6 * len(filenames)
    for name in filenames[:int(i)]:
        f.write(os.path.join(path, name) + ' 1\n')
    other = filenames[int(i):]
    i = 0.5 * len(other)
    for name in other[:int(i)]:
        c.write(os.path.join(path, name) + ' 1\n')
    for name in other[int(i):]:
        t.write(os.path.join(path, name) + ' 1\n')
for (path, dirnames, filenames) in os.walk('data/fear'):
    i = 0.6 * len(filenames)
    for name in filenames[:int(i)]:
        f.write(os.path.join(path, name) + ' 2\n')
    other = filenames[int(i):]
    i = 0.5 * len(other)
    for name in other[:int(i)]:
        c.write(os.path.join(path, name) + ' 2\n')
    for name in other[int(i):]:
        t.write(os.path.join(path, name) + ' 2\n')
for (path, dirnames, filenames) in os.walk('data/happy'):
    i = 0.6 * len(filenames)
    for name in filenames[:int(i)]:
        f.write(os.path.join(path, name) + ' 3\n')
    other = filenames[int(i):]
    i = 0.5 * len(other)
    for name in other[:int(i)]:
        c.write(os.path.join(path, name) + ' 3\n')
    for name in other[int(i):]:
        t.write(os.path.join(path, name) + ' 3\n')
for (path, dirnames, filenames) in os.walk('data/sad'):
    i = 0.6 * len(filenames)
    for name in filenames[:int(i)]:
        f.write(os.path.join(path, name) + ' 4\n')
    other = filenames[int(i):]
    i = 0.5 * len(other)
    for name in other[:int(i)]:
        c.write(os.path.join(path, name) + ' 4\n')
    for name in other[int(i):]:
        t.write(os.path.join(path, name) + ' 4\n')
for (path, dirnames, filenames) in os.walk('data/surprise'):
    i = 0.6 * len(filenames)
    for name in filenames[:int(i)]:
        f.write(os.path.join(path, name) + ' 5\n')
    other = filenames[int(i):]
    i = 0.5 * len(other)
    for name in other[:int(i)]:
        c.write(os.path.join(path, name) + ' 5\n')
    for name in other[int(i):]:
        t.write(os.path.join(path, name) + ' 5\n')
for (path, dirnames, filenames) in os.walk('data/neutral'):
    i = 0.6 * len(filenames)
    for name in filenames[:int(i)]:
        f.write(os.path.join(path, name) + ' 6\n')
    other = filenames[int(i):]
    i = 0.5 * len(other)
    for name in other[:int(i)]:
        c.write(os.path.join(path, name) + ' 6\n')
    for name in other[int(i):]:
        t.write(os.path.join(path, name) + ' 6\n')
