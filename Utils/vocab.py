# 词典：进行词的映射

import collections


def count_corpus(tokens):
    """统计词频

    Args:
        tokens (array): 词元列表
    """
    # tokens 可能是1D or 2D
    if len(tokens) == 0 or isinstance(tokens[0],list):
        tokens = [token for line in tokens for token in line]   # 碾平列表
    return collections.Counter(tokens)                           # 统计词频


class Vocab:                    
    """按次序来的
    """
    
    def __init__(self,tokens=None,min_fred=0,reserved_tokens=None) -> None:
        """初始化类

        Args:
            tokens (_type_, optional): 进行词元化后的数据. Defaults to None.
            min_fred (int, optional): 词的最小频次，小于该值就将其去掉. Defaults to 0.
            reserved_tokens (_type_, optional): 保留的符号，pad等. Defaults to None.
        """
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率进行排序
        counter = count_corpus(tokens)
        # 对counter进行排序
        self._token_freqs = sorted(counter.items(),key=lambda x:x[1],reverse=True)
        
        # 未知词元的索引是0
        self.idx_to_token = ['<unk>']+reserved_tokens                                   # 带有unk的未知词元列表
        self.token_to_idx = {token:idx for idx ,token in enumerate(self.idx_to_token)}  #{'<unk>':0,之后出现的单词:单词的索引}
        
        
        # 在这里会扫描碾平的词元列表，将其添加到unk的未知词元列表idx_to_token
        for token,freq in self._token_freqs:
            if freq < min_fred:     # 当一个词出现的频率小于min_fred的时候，就退出循环
                break   # 退出当次循环
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)         # 将没有出现过的词元放入idx_to_token
                self.token_to_idx[token] = len(self.idx_to_token)-1     # 将新添加的词加上索引,因为之前已经有数据（'unk'等）了，所以要减一
                
    # 运算符重载，使用len可以返回长度
    def __len__(self):
        return len(self.idx_to_token)
    
    # 运算符重载，通过字符返回其字典对应的值
    def __getitem__(self,tokens):  
        """取出tokens里面每个字符的索引
        这是一个递归函数，如果传入的是列表或者元素，会递归调用函数，取出里面元素对应的索引

        """
        if not isinstance(tokens,(list,tuple)):             # 如果tokens不是列表或字典
            return self.token_to_idx.get(tokens,self.unk)   # 从为未知词元里面去寻找，找不到返回unk    
        return [self.__getitem__(token) for token in tokens]    # 返回token
    
    # 给定索引，返回idx_to_token的元素
    def to_tokens(self,indices):
        if not isinstance(indices,(list,tuple)):
            return self.idx_to_token[indices]       # 返回未知词元
        return [self.idx_to_token[index] for index in indices]

    @property        
    def unk(self):      # 未知词元索引为0
        return 0
    @property
    def token_freqs(self):
        return self._token_freqs    
