import os

labels = os.listdir('./data')
for label in labels:
    la_dir = os.path.join('./data', label)
    files = os.listdir(la_dir)
    no = 0
    for file in files:
        ori = os.path.join(la_dir, file)
        tar = str(no) + '.jpg'
        tmp = os.popen('ren {0} {1}'.format(os.path.abspath(ori), tar))
        no += 1
