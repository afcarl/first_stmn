# first_working_stmn
```
mkdir data
cd data

wget http://www.cs.toronto.edu/~rkiros/models/dictionary.txt
wget http://www.cs.toronto.edu/~rkiros/models/utable.npy
wget http://www.cs.toronto.edu/~rkiros/models/btable.npy
wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz
wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz.pkl
wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz
wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz.pkl
```

```
mkdir en
cd en

http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz
```
```
python wmemnn.py en/qa5_three-arg-relations_train.txt
python wmemnn.py en/qa5_short_train.txt
```

