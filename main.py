from functools import reduce

x = 5,6,7
y = 3,6,10

x = 17,28,30
y = 99,16,8

def vote(acc, vals):
    left_score, right_score = acc
    left_val, right_val = vals
    if left_val == right_val: return acc
    return [left_score+1, right_score] if left_val > right_val else [left_score, right_score+1]



print(reduce(vote , zip(x,y), [0, 0]))

from algorithms import KMC, KNN

def runner():
    knn = KNN([2,3], [
        ['lol', 2,3], 
        ['lol', 4,5,],
        [ 'no', 1,3],
        ['hah', 3,7],
        ['hey', 3,3,]
    ], 3)

    kmc = KMC([
        [2,3], 
        [4,5],
        [1,3],
        [3,7]
    ], 2)

    return knn, kmc


print(runner())