from  hanlp import hanlp
import jieba
import torch
import numpy as np
import pandas as pd
from . import Trie,vocab
import re
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from d2l import torch as d2l

def got_chinese(x):
    """保留中文

    Args:
        x (str): sentence

    Returns:
        str: sentence
    """
    # 定义正则表达式
    pattern = re.compile(r'[\u4e00-\u9fa5]')

    # 使用正则表达式查找匹配文本
    match = pattern.findall(x)

    # 输出匹配结果
    return ''.join(match)


def cut_word(x,method='hanlp'):
    """分词
    
    Args:
        x (list): 列表中存储着分词的句子
        method (str, optional): 使用jieba分词或者hanlp分词(更强大的分词，多种模型可供选择). Defaults to 'hanlp'.

    Returns:
        list[list]: 嵌套列表，内部是分词好的句子
    """
    if method == 'jieba':
        return [jieba.lcut(i) for i in x]
    elif method == 'hanlp':
        tok = hanlp.load(hanlp.pretrained.tok.MSR_TOK_ELECTRA_BASE_CRF)  # Electra ( Clark et al. 2020 ) 在 MSR CWS 数据集上训练的基础模型。性能远高于MTL机型P: 98.71% R: 98.64%    
        return tok(x)


        
def Text2Vec(x,method='word2vec'):
    """预训练模型表示法

    Args:
        x (list[list]): [['tok1','tok2',...],['tok1','tok2'],...]
        method (str, optional): _description_. Defaults to 'word2vec'.
    """
    if method=='word2vec':
        model = hanlp.load(hanlp.pretrained.word2vec.MERGE_SGNS_BIGRAM_CHAR_300_ZH)
        res = []
        for i in x:
            res.append([model(j).cpu() for j in i])
        res = [np.sum(i)/len(i) for i in res]               # 因为句子的长度不一，在这里进行取个均值的操作（损失信息）。如何可以的话可以进行拼接tensor
        return res
    elif method=='glov':    # 全局词向量
        model = hanlp.load(hanlp.pretrained.glove.GLOVE_6B_300D)
        res = []
        for i in x:
            res.append([model(j).cpu() for j in i])
        res = [np.sum(i)/len(i) for i in res]               # 因为句子的长度不一，在这里进行取个均值的操作（损失信息）。如何可以的话可以进行拼接tensor
        return res
    elif method == 'fasttxt':
        model = hanlp.load(hanlp.pretrained.fasttext.FASTTEXT_CC_300_EN)
        res = []
        for i in x:
            res.append([model(j).cpu() for j in i])
        res = [np.sum(i)/len(i) for i in res]               # 因为句子的长度不一，在这里进行取个均值的操作（损失信息）。如何可以的话可以进行拼接tensor
        return res


def Dropwords(drop_words,sentence_array):
    import gc
    """基于字典树的高效查找与搜索

    `注意：如果是中文的词过多的话，这个数据结构可能会占用大量的内存！！！！`
    Args:
        drop_words (_type_): _description_
        sentence_array (_type_): _description_

    Returns:
        _type_: _description_
    """
    t = Trie.Trie()
    for i in drop_words:
        t.insert(i)
    
    contents_clean = []
    all_words = []
    for line in sentence_array:
        line_clean = []
        for word in line:
            if t.search(word):
                continue
            line_clean.append(word)
            all_words.append(str(word))
        contents_clean.append(line_clean)
    del t
    gc.collect()
    return contents_clean,all_words



def truncate_pad(line, num_steps, padding_token):
    """截取或者填充句子

    Args:
        line (array): 一个句子
        num_steps (int): 时间步数，也就是我们所说的长度
        padding_token (str): 填充的token（词元）

    Returns:
        list: list
    """
    if len(line) > num_steps:
        return line[:num_steps]  # 截取
    return line + [padding_token] * (num_steps - len(line))  # 填充


# 小批量加载
class Dataset(torch.utils.data.Dataset):             # pytorch对数据的封装，重写此方法

    def __init__(self, data):
        """初始化

        Args:
            data (array[array]): 传入这样形式的数据：[[features,label],[features,label],...]
        """
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i]

def collate_fn(data):               # 传入的数据，对数据的输出形式进行设定
    inputs = [torch.tensor(i[0]) for i in data]
    lengths = torch.tensor([len(i[0]) for i in data])
    targets = torch.tensor([i[1] for i in data], dtype=torch.long)
    return torch.stack(inputs), lengths, targets


def load_data(
    batch_size,
    num_steps=25,
    min_fred = 0,
    path="../Get_Data/train2.csv",
    collate_fn=collate_fn,
    method = "hanlp",
    is_truncate_pad=True,
    is_dropword=False,
    is_gotchinese=False ):
                            
    """加载数据

    Args:
        batch_size (_type_): _description_
        num_steps (int, optional): _description_. Defaults to 25.
        min_fred (int, optional): vocab的删掉频率低于阈值的词. Defaults to 0.
        path (str, optional): _description_. Defaults to "../Get_Data/train2.csv".
        collate_fn (_type_, optional): _description_. Defaults to collate_fn.
        method (str, optional): _description_. Defaults to "hanlp".
        is_truncate_pad (bool, optional): _description_. Defaults to True.
        is_dropword (bool, optional): _description_. Defaults to False.
        is_gotchinese (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    data = pd.read_csv(path)
    if is_gotchinese:
        data['title'] = data['title'].map(got_chinese)     # 取出标点符号
    cut_words = cut_word(data['title'].to_list(),method=method)  # 分词
    stopwords =  pd.read_csv("../Get_data/hit_stopwords.txt",index_col=False,sep="\t",quoting=3,names=['stopword'], encoding='utf-8').stopword.values.tolist() #list
    cut_words = Dropwords(stopwords,cut_words)[0] if is_dropword else cut_words
    vocab_ = vocab.Vocab(cut_words,reserved_tokens=['<pad>'],min_fred=min_fred)
    X = [truncate_pad(vocab_[i],num_steps,vocab_['<pad>']) for i in cut_words] if is_truncate_pad else [vocab_[i] for i in cut_words]
    y = data['label']
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    x_train, x_test, y_train, y_test = train_test_split(X, y,test_size=0.2 ,shuffle=True ,random_state=1,stratify=y)
    train_dataset = Dataset(list(zip(x_train,y_train)))
    test_dataset = Dataset(list(zip(x_test,y_test)))
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)
    return train_data_loader,test_data_loader,vocab_,encoder


def grad_clipping(net, theta):  
    """裁剪梯度

    Args:
        net (net): 网络架构
        theta (float): 裁剪阈值
    """
    if isinstance(net, torch.nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
            

def TokenEmbedding(x,method='word2vec'):
    """Token to embedding

    Args:
        x (_type_): idx_to_token,字符
        method (str, optional): _description_. Defaults to 'word2vec'.

    Returns:
        _type_: token对应的词向量
    """
    if method=='word2vec':
        model = hanlp.load(hanlp.pretrained.word2vec.MERGE_SGNS_BIGRAM_CHAR_300_ZH)
        return model(x)
    elif method=='glov':    # 全局词向量,以为是中文的，没想到是英文的。第三方词向量，无法使用
        model = hanlp.load(hanlp.pretrained.glove.GLOVE_6B_300D)
        return model(x)
    elif method == 'fasttxt':
        model = hanlp.load(hanlp.pretrained.fasttext.FASTTEXT_CC_300_EN)
        return model(x)
    
    
# 封装成函数,机器学习数据加载函数
def load_data2(test_size=0.2,path = "../Get_Data/train2.csv",is_gotchinses=False,method='hanlp'):
    """加载数据

    Args:
        test_size (float, optional): 测试集比例. Defaults to 0.2.
        path (str, optional): 文件路径. Defaults to "../Get_Data/train.csv".

    Returns:
        _type_: _description_
    """
    data = pd.read_csv(path)
    if is_gotchinses:
        data['title'] = data['title'].map(got_chinese)        # 除去标点符号
        
    cut_words = cut_word(data['title'].to_list(),method=method)  # 进行分词
    stopwords =  pd.read_csv("../Get_data/hit_stopwords.txt",index_col=False,sep="\t",quoting=3,names=['stopword'], encoding='utf-8').stopword.values.tolist() #list
    cut_words = Dropwords(stopwords,cut_words)[0] 
    data['title'] = cut_words
    data.title = data.title.apply(lambda x:" ".join(x))     # 出去中括号
    encoder = LabelEncoder()
    data['label'] = encoder.fit_transform(data['label'])    # 标签映射
    x_train, x_test, y_train, y_test = train_test_split(data['title'].values, data['label'].values, test_size=0.2,random_state=1,shuffle=True,stratify=data['label'])
    return x_train,y_train,x_test,y_test,cut_words,encoder

def pre_data(x_train_,x_test_,n = 0.95,method='count',is_pca=True):
    """数据处理

    Args:
        x_train_ (_type_): _description_
        x_test_ (_type_): _description_
        n (float, optional): 保留信息的数量. Defaults to 0.95.
        method (str, optional): 文本表示方法. Defaults to 'count'.
        is_pca (bool, optional): 是否降维. Defaults to True.

    Returns:
        _type_: _description_
    """
    if method == "count":
        COUNT = CountVectorizer()
        x_train_ = COUNT.fit_transform(x_train_).toarray()        # 转换训练集
        x_test_ = COUNT.transform(x_test_).toarray()             # 假设测试集与训练集同分布，进行Transformer转换
        pca = PCA(n_components=n)                # 保留95%的信息
        if is_pca:
            x_train_ = pca.fit_transform(x_train_)
            x_test_ = pca.transform(x_test_)
            return x_train_,x_test_   # 假设同分布，先进行fit
        return x_train_,x_test_   
    else:
        TFIDF = TfidfVectorizer()
        x_train_ = TFIDF.fit_transform(x_train_).toarray()        # 转换训练集
        x_test_ = TFIDF.transform(x_test_).toarray()              # 假设测试集与训练集同分布，进行Transformer转换
        pca = PCA(n_components=n)                # 保留95%的信息
        if is_pca:
            x_train_ = pca.fit_transform(x_train_)
            x_test_ = pca.transform(x_test_)
            return x_train_,x_test_   # 假设同分布，先进行fit
        return x_train_,x_test_   
    

class Accumulator:
    """计数器"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
def accuracy(y_hat, y):  
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)            # 取出预测值
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def acc(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)            # 取出预测值
    
    return float(sum(y_hat == y))

def evaluate_acc(net, data_iter, device=None):
    """评估测试集

    Args:
        net (_type_): _description_
        data_iter (_type_): _description_
        device (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if isinstance(net, torch.nn.Module):
        net.eval()  
        if not device:
            device = next(iter(net.parameters())).device

    metric = Accumulator(2)

    with torch.no_grad():
        for X, length, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(acc(net(X,length), y), y.numel())
    return metric[0] / metric[1]

def train_batch_with_length(net, X, length, y, loss, optimizer, device):
    """传入句子长度batch训练

    Args:
        net (_type_): _description_
        X (_type_): _description_
        length (_type_): _description_
        y (_type_): _description_
        loss (_type_): _description_
        optimizer (_type_): _description_
        device (_type_): _description_

    Returns:
        _type_: _description_
    """
    if isinstance(X, list):
        X = [x.to(device) for x in X]
    else:
        X = X.to(device)
    y = y.to(device)
    net.train()
    optimizer.zero_grad()
    pred = net(X,length)
    l = loss(pred, y)
    l.sum().backward()
    optimizer.step()
    train_loss_sum = l.sum()
    train_acc_sum = acc(pred,y)
    return train_loss_sum, train_acc_sum

def train_with_length(net, train_iter, test_iter, loss, optimizer, num_epochs,
               device,scheduler=None):
    """传入句子长度的训练
    acc始终是累加的，每个epoch结束，输出当前正确率

    Args:
        net (_type_): _description_
        train_iter (_type_): _description_
        test_iter (_type_): _description_
        loss (_type_): _description_
        optimizer (_type_): _description_
        num_epochs (_type_): _description_
        device (_type_, optional): _description_. Defaults to device.
    """
    train_acc_list = []
    test_acc_list = []
    train_loss_list = []
    num_batches = len(train_iter)

    net.to(device)
    for epoch in range(num_epochs):
        print("-"*65)
        metric = Accumulator(4)
        for i, (features, length, labels) in enumerate(train_iter):
            l, acc = train_batch_with_length(
                net, features, length, labels, loss, optimizer, device)
            metric.add(l, acc, labels.shape[0], labels.numel())
            
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                
                print(f"| epoch{epoch:3d} | avg_loss:{metric[0] / metric[2]:.3f} | train_acc:{metric[1] / metric[3]:.2f} |")
                
        if scheduler:
            scheduler.step()                                # 进行warm up
        print('-'*65)      
        test_acc = evaluate_acc(net, test_iter)
        print(f"| epoch{epoch:3d} | avg_loss:{metric[0] / metric[2]:.3f} | train_acc:{metric[1] / metric[3]:.2f} | test_acc:{test_acc:.2f}| lr:{optimizer.param_groups[0]['lr']} | ")
        train_acc_list.append(metric[1] / metric[3])
        train_loss_list.append(metric[0] / metric[2])
        test_acc_list.append(test_acc)
        
    print("-"*89)   
    print("final metris:")
    print(f'| avg_loss {metric[0] / metric[2]:.3f} | train acc '
          f'{metric[1] / metric[3]:.3f} | test acc {test_acc:.3f} |')
    train_acc_list.append(metric[1] / metric[3])
    train_loss_list.append(metric[0] / metric[2])
    test_acc_list.append(test_acc)
    plt.plot(train_acc_list,label="train_acc")
    plt.plot(train_loss_list,label="train_loss")
    plt.plot(test_acc_list,label="test_acc")
    plt.legend()
    plt.show()


def train(net, train_iter, test_iter, loss, trainer, num_epochs,
                devices=d2l.try_all_gpus(),scheduler=None):
    """改写李沫train_ch13()训练方法
    加入warm-up

    Args:
        net (_type_): _description_
        train_iter (_type_): _description_
        test_iter (_type_): _description_
        loss (_type_): _description_
        trainer (_type_): _description_
        num_epochs (_type_): _description_
        devices (_type_, optional): _description_. Defaults to d2l.try_all_gpus().
        scheduler (_type_, optional): _description_. Defaults to None.
    """
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = torch.nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):

        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = d2l.train_batch_ch13(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                            (metric[0] / metric[2], metric[1] / metric[3],
                            None))
        if scheduler:
            scheduler.step()
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
        f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
        f'{str(devices)}')