# 字典树，高效查找与搜索
class Trie:
    def __init__(self):
        self.child = {}

    def insert(self, word):
        a = self.child
        for i in word:
            if not i in a.keys():
                a[i] = {}
            a = a[i]
        a["end"] = True
    
    def search(self, word):
        a = self.child
        for i in word:
            if not i in a.keys():
                return False
            a = a[i]
        return True if "end" in a.keys() else False
    
    def startsWith(self, prefix):
        a = self.child
        for i in prefix:
            if not i in a.keys():
                return False
            a = a[i]
        return True