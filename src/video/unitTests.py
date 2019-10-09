from video.getShotMomentsNew import findLastIndex

def test_findLastIndex():
    myList = [0, 0, 0.1, 0, 0.05, 0]
    index = findLastIndex(myList, condition = lambda v : v>0)

    print(index)

if __name__ == '__main__':
    test_findLastIndex()