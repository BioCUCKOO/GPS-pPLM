import sys, os, re, joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from collections import Counter, OrderedDict
import torch.nn.functional as F
import gc
import math
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import roc_auc_score
import json

def fasta2seq(filename, site):
    fas = open(filename, 'r')
    dict_id2fast = {}
    global linename

    for line in fas:
        if line.startswith('>'):
            linename = line.replace('>', '').strip()
            dict_id2fast[linename] = ''
        else:
            dict_id2fast[linename] += line.strip()

    unid_list = []
    postion_list = []
    seq_list = []
    for unid, fasta in dict_id2fast.items():
        seqs = []
        lenth = len(fasta)
        if site == 'ST':
            for index in range(len(fasta)):
                if fasta[index] == 'S' or fasta[index] == 'T':

                    if index < 10:
                        seqs.append((10 - index) * '*' + fasta[:index + 11])

                    elif lenth - index - 1 < 10:
                        seqs.append(fasta[index - 10:] + '*' * (index + 11 - lenth))

                    else:
                        seqs.append(fasta[index - 10:index + 11])
                    postion_list.append(index)
        seq_list.extend(seqs)

        unid_list.extend([unid] * len(seqs))

    return dict_id2fast, unid_list, seq_list, postion_list

def Protein_APAAC(seq_list):
    lambdaValue = 2
    w = 0.05
    with open('PAAC.txt','r') as f:
        records = f.readlines()
    AA = ''.join(records[0].rstrip().split()[1:])
    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i
    AAProperty = []
    AAPropertyNames = []
    for i in range(1, len(records) - 1):
        array = records[i].rstrip().split() if records[i].rstrip() != '' else None
        AAProperty.append([float(j) for j in array[1:]])
        AAPropertyNames.append(array[0])

    AAProperty1 = []
    for i in AAProperty:
        meanI = sum(i) / 20
        fenmu = math.sqrt(sum([(j - meanI) ** 2 for j in i]) / 20)
        AAProperty1.append([(j - meanI) / fenmu for j in i])
    dict_feature = {}
    for seq in seq_list:
        name = seq
        seq = seq.replace('X', '*').replace('U', '*').replace('O', '*').replace('*', '')
        sequence = seq
        feature = []

        theta = []
        for n in range(1, lambdaValue + 1):
            for j in range(len(AAProperty1)): theta.append(sum(
                [AAProperty1[j][AADict[sequence[k]]] * AAProperty1[j][AADict[sequence[k + n]]] for k in
                 range(len(sequence) - n)]) / (len(sequence) - n))
        myDict = {}
        for aa in AA:
            myDict[aa] = sequence.count(aa)

        feature = feature + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
        feature = feature + [w * value / (1 + w * sum(theta)) for value in theta]

        newfeature = [round(float(i), 6) for i in feature]
        dict_feature[name] = newfeature

    return dict_feature

def get_feature_num(input, k):
    new_input = input.replace('*', '')
    lenth = len(new_input)
    dict_CKSAAP = {}
    pos = 0
    for i in range(lenth - k - 1):
        AA = new_input[pos] + new_input[pos + k + 1]
        if AA not in dict_CKSAAP:
            dict_CKSAAP[AA] = 1
        else:
            dict_CKSAAP[AA] += 1
        pos += 1
    return dict_CKSAAP

def get_AApos():
    AA_str = 'ACDEFGHIKLMNPQRSTVWY'
    dict_AApos = {}
    pos = 0
    num = 0
    for j in range(20):
        for i in range(20):
            dict_AApos[AA_str[pos] + AA_str[i]] = num
            num += 1
        pos += 1
    return dict_AApos

def get_CSKAAP_feature(input, k):
    new_input = input.replace('*', '')
    lenth = len(new_input)

    dict_CKSAAP = get_feature_num(input, k)

    dict_AApos = get_AApos()
    feature = [0] * 400

    for key in dict_CKSAAP:
        feature_num = dict_CKSAAP[key] / (lenth - k - 1)
        if key not in dict_AApos:
            continue
        position = dict_AApos[key]
        feature[position] = feature_num

    return feature

def get_from_file(seq_list):
    dict_feature = {}
    for seq in seq_list:
        name = seq
        feature = get_CSKAAP_feature(seq, 0) + get_CSKAAP_feature(seq, 1) + get_CSKAAP_feature(seq, 2) + get_CSKAAP_feature(
            seq, 3)
        newfeature = [round(float(i), 9) for i in feature]
        dict_feature[name] = newfeature

    return dict_feature

def Count(seq1, seq2):
    sum = 0
    for aa in seq1:
        sum = sum + seq2.count(aa)
    return sum

def Protein_CTDC(seq_list):
    dict_feature = {}
    group1 = {
        'hydrophobicity_PRAM900101': 'RKEDQN',
        'hydrophobicity_ARGP820101': 'QSTNGDE',
        'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
        'hydrophobicity_PONP930101': 'KPDESNQT',
        'hydrophobicity_CASG920101': 'KDEQPSRNTG',
        'hydrophobicity_ENGD860101': 'RDKENQHYP',
        'hydrophobicity_FASG890101': 'KERSQD',
        'normwaalsvolume': 'GASTPDC',
        'polarity': 'LIFWCMVY',
        'polarizability': 'GASDT',
        'charge': 'KR',
        'secondarystruct': 'EALMQKRH',
        'solventaccess': 'ALFCGIVW'
    }
    group2 = {
        'hydrophobicity_PRAM900101': 'GASTPHY',
        'hydrophobicity_ARGP820101': 'RAHCKMV',
        'hydrophobicity_ZIMJ680101': 'HMCKV',
        'hydrophobicity_PONP930101': 'GRHA',
        'hydrophobicity_CASG920101': 'AHYMLV',
        'hydrophobicity_ENGD860101': 'SGTAW',
        'hydrophobicity_FASG890101': 'NTPG',
        'normwaalsvolume': 'NVEQIL',
        'polarity': 'PATGS',
        'polarizability': 'CPNVEQIL',
        'charge': 'ANCQGHILMFPSTWYV',
        'secondarystruct': 'VIYCWFT',
        'solventaccess': 'RKQEND'
    }
    group3 = {
        'hydrophobicity_PRAM900101': 'CLVIMFW',
        'hydrophobicity_ARGP820101': 'LYPFIW',
        'hydrophobicity_ZIMJ680101': 'LPFYI',
        'hydrophobicity_PONP930101': 'YMFWLCVI',
        'hydrophobicity_CASG920101': 'FIWC',
        'hydrophobicity_ENGD860101': 'CVLIMF',
        'hydrophobicity_FASG890101': 'AYHWVMFLIC',
        'normwaalsvolume': 'MHKFRYW',
        'polarity': 'HQRKNED',
        'polarizability': 'KMHFRYW',
        'charge': 'DE',
        'secondarystruct': 'GNPSD',
        'solventaccess': 'MSPTHY'
    }

    property = (
        'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101',
        'hydrophobicity_PONP930101',
        'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
        'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')
    for seq in seq_list:
        name = seq
        sequence = seq
        feature = []
        for p in property:
            c1 = Count(group1[p], sequence) / len(sequence)
            c2 = Count(group2[p], sequence) / len(sequence)
            c3 = 1 - c1 - c2
            feature = feature + [c1, c2, c3]
        newfeature = [round(float(i), 6) for i in feature]
        dict_feature[name] = newfeature

    return dict_feature

def Protein_DDE(seq_list):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    dict_feature = {}
    myCodons = {'A': 4, 'C': 2, 'D': 2, 'E': 2, 'F': 2, 'G': 4, 'H': 2, 'I': 3, 'K': 2, 'L': 6,
                'M': 1, 'N': 2, 'P': 4, 'Q': 2, 'R': 6, 'S': 6, 'T': 4, 'V': 4, 'W': 1, 'Y': 2
                }

    diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]

    myTM = []
    for pair in diPeptides:
        myTM.append((myCodons[pair[0]] / 61) * (myCodons[pair[1]] / 61))

    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i
    for seq in seq_list:
        name = seq
        seq = seq.replace('X', '*').replace('U', '*').replace('O', '*').replace('*', '')
        sequence = seq
        feature = []

        tmpCode = [0] * 400
        for j in range(len(sequence) - 2 + 1):
            tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j + 1]]] = tmpCode[AADict[sequence[j]] * 20 + AADict[
                sequence[j + 1]]] + 1
        if sum(tmpCode) != 0:
            tmpCode = [i / sum(tmpCode) for i in tmpCode]

        myTV = []
        for j in range(len(myTM)):
            myTV.append(myTM[j] * (1 - myTM[j]) / (len(sequence) - 1))

        for j in range(len(tmpCode)):
            tmpCode[j] = (tmpCode[j] - myTM[j]) / math.sqrt(myTV[j])

        feature = feature + tmpCode
        newfeature = [round(float(i), 6) for i in feature]
        dict_feature[name] = newfeature

    return dict_feature

def Protein_DistancePair(seq_list):
    dict_feature = {}
    cp20_dict = {
        'A': 'A',
        'C': 'C',
        'D': 'D',
        'E': 'E',
        'F': 'F',
        'G': 'G',
        'H': 'H',
        'I': 'I',
        'K': 'K',
        'L': 'L',
        'M': 'M',
        'N': 'N',
        'P': 'P',
        'Q': 'Q',
        'R': 'R',
        'S': 'S',
        'T': 'T',
        'V': 'V',
        'W': 'W',
        'Y': 'Y',
    }
    cp19_dict = {
        'A': 'A',
        'C': 'C',
        'D': 'D',
        'E': 'E',
        'F': 'F',
        'G': 'G',
        'H': 'H',
        'I': 'I',
        'K': 'K',
        'L': 'L',
        'M': 'M',
        'N': 'N',
        'P': 'P',
        'Q': 'Q',
        'R': 'R',
        'S': 'S',
        'T': 'T',
        'V': 'V',
        'W': 'W',
        'Y': 'F',
    }
    cp14_dict = {
        'A': 'A',
        'C': 'C',
        'D': 'D',
        'E': 'E',
        'F': 'F',
        'G': 'G',
        'H': 'H',
        'I': 'I',
        'K': 'H',
        'L': 'L',
        'M': 'I',
        'N': 'N',
        'P': 'P',
        'Q': 'H',
        'R': 'H',
        'S': 'S',
        'T': 'T',
        'V': 'I',
        'W': 'W',
        'Y': 'W',
    }
    cp13_dict = {
        'A': 'A',
        'C': 'C',
        'D': 'D',
        'E': 'E',
        'F': 'F',
        'G': 'G',
        'H': 'H',
        'I': 'I',
        'K': 'K',
        'L': 'I',
        'M': 'F',
        'N': 'N',
        'P': 'H',
        'Q': 'H',
        'R': 'K',
        'S': 'S',
        'T': 'T',
        'V': 'V',
        'W': 'H',
        'Y': 'H',
    }

    cp20_AA = 'ACDEFGHIKLMNPQRSTVWY'
    cp19_AA = 'ACDEFGHIKLMNPQRSTVW'
    cp14_AA = 'ACDEFGHILNPSTW'
    cp13_AA = 'ACDEFGHIKNSTV'

    distance = 0
    cp = 'cp(20)'

    AA = cp20_AA
    AA_dict = cp20_dict
    if cp == 'cp(19)':
        AA = cp19_AA
        AA_dict = cp19_dict
    if cp == 'cp(14)':
        AA = cp14_AA
        AA_dict = cp14_dict
    if cp == 'cp(13)':
        AA = cp13_AA
        AA_dict = cp13_dict

    pair_dict = {}
    single_dict = {}
    for aa1 in AA:
        single_dict[aa1] = 0
        for aa2 in AA:
            pair_dict[aa1 + aa2] = 0

    # clear
    for seq in seq_list:
        name = seq
        seq = seq.replace('X', '*').replace('U', '*').replace('O', '*').replace('*', '')

        sequence = seq
        feature = []

        for d in range(distance + 1):
            if d == 0:
                tmp_dict = single_dict.copy()
                for i in range(len(sequence)):
                    tmp_dict[AA_dict[sequence[i]]] += 1
                for key in sorted(tmp_dict):
                    feature.append(tmp_dict[key] / len(sequence))
            else:
                tmp_dict = pair_dict.copy()
                for i in range(len(sequence) - d):
                    tmp_dict[AA_dict[sequence[i]] + AA_dict[sequence[i + d]]] += 1
                for key in sorted(tmp_dict):
                    feature.append(tmp_dict[key] / (len(sequence) - d))

            newfeature = [round(float(i), 6) for i in feature]
            dict_feature[name] = newfeature

    return dict_feature

def Protein_EAAC(seq_list):
    dict_feature = {}
    kw = 5
    AA = 'ARNDCQEGHILKMFPSTWYV'
    for seq in seq_list:
        name = seq
        sequence = seq
        feature = []

        for j in range(len(sequence)):
            if j < len(sequence) and j + kw <= len(sequence):
                count = Counter(sequence[j:j + kw])
                for key in count:
                    count[key] = count[key] / len(sequence[j:j + kw])
                for aa in AA:
                    feature.append(count[aa])

        newfeature = [round(float(i), 6) for i in feature]
        dict_feature[name] = newfeature

    return dict_feature

def ESM(seq_list):
    dict_feature = {}
    model, alphabet = torch.hub.load("torch/hub/facebookresearch_esm_main", "esm2_t33_650M_UR50D", source='local')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    for seq in seq_list:
        name = seq
        seq_qu = seq.replace('*', '')
        data = [(f"{seq}", seq_qu)]

        batch_converter = alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33])
            token_representations = results["representations"][33]
        for i, (_, seq) in enumerate(data):
            seq_len = len(seq) + 2
            newfeature = [float(i) for i in token_representations[i, 1:seq_len - 1].mean(0).tolist()]
        dict_feature[name] = newfeature
    return dict_feature

def Rvalue(aa1, aa2, AADict, Matrix):
    return sum([(Matrix[i][AADict[aa1]] - Matrix[i][AADict[aa2]]) ** 2 for i in range(len(Matrix))]) / len(Matrix)

def Protein_PAAC(seq_list):
    dict_feature = {}
    lambdaValue = 2
    w = 0.05
    dataFile = 'PAAC.txt'
    with open(dataFile) as f:
        records = f.readlines()
    AA = ''.join(records[0].rstrip().split()[1:])
    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i
    AAProperty = []
    AAPropertyNames = []
    for i in range(1, len(records)):
        array = records[i].rstrip().split() if records[i].rstrip() != '' else None
        AAProperty.append([float(j) for j in array[1:]])
        AAPropertyNames.append(array[0])

    AAProperty1 = []
    for i in AAProperty:
        meanI = sum(i) / 20
        fenmu = math.sqrt(sum([(j - meanI) ** 2 for j in i]) / 20)
        AAProperty1.append([(j - meanI) / fenmu for j in i])
    for seq in seq_list:
        name = seq
        seq = seq.replace('X', '*').replace('U', '*').replace('O', '*').replace('*', '')
        sequence = seq
        feature = []

        theta = []
        for n in range(1, lambdaValue + 1):
            theta.append(
                sum([Rvalue(sequence[j], sequence[j + n], AADict, AAProperty1) for j in range(len(sequence) - n)]) / (
                            len(sequence) - n))
        myDict = {}
        for aa in AA:
            myDict[aa] = sequence.count(aa)
        feature = feature + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
        feature = feature + [(w * j) / (1 + w * sum(theta)) for j in theta]

        newfeature = [round(float(i), 6) for i in feature]
        dict_feature[name] = newfeature
    return dict_feature

def readweight(weight_file):
    weight = None
    with open(weight_file, 'r') as f:
        for i, line in enumerate(f):
            if i == 2 - 1:
                weight = np.array([float(x) for x in line.rstrip().split('\t')])
    return weight

def readPeptide(pepfile, lr):
    data = []
    lr = 10 - lr
    with open(pepfile, 'r') as f:
        for line in f:
            if lr == 0:
                data.append(line.rstrip().split('\t')[0].replace('X','*'))
            else:
                data.append(line.rstrip().split('\t')[0].replace('X','*'))
    return data

def gps(seq_list):
    global gpn
    dict_feature = {}
    def generateMMData(querylist, plist, pls_weight, mm_weight, loo=True, positive=False):
        gp = GpsPredictor(plist, pls_weight, mm_weight)
        d = []
        for query_peptide in querylist:
            d.append(gp.generateMMdata(query_peptide, loo).tolist())
        return d

    def generatePLSdata(querylist, plist, pls_weight, mm_weight, loo=True, positive=False):
        gp = GpsPredictor(plist, pls_weight, mm_weight)
        d = []
        for query_peptide in querylist:
            d.append(gp.generatePLSdata(query_peptide, loo).tolist())
        return d

    mm_weight = readweight('BLOSUM62R.txt')
    ll = len(seq_list[0])
    plist = readPeptide('positive.txt', int(ll / 2))
    gpn1 = generateMMData(seq_list, plist, np.repeat(1, ll), mm_weight, loo=False, positive=False)

    for i in range(len(gpn1)):
        dict_feature[seq_list[i]] = gpn1[i]

    return dict_feature

class GpsPredictor(object):
    def __init__(self, plist, pls_weight, mm_weight):
        '''
        initial GPS predictor using positive training set, pls_weight vector and mm_weight vector
        :param plist: (list) positive peptides list
        :param pls_weight:  (list) pls_weight vector
        :param mm_weight:   (list) mm_weight vector
        '''
        self.alist = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y',
                      'V', '*']
        self.plist = plist
        self.pls_weight = np.array(pls_weight).flatten()
        self.mm_weight = np.array(mm_weight).flatten()


        if os.path.exists('mm_matrix.npy') and os.path.exists('mm_intercept.npy'):
            self.__mm_matrix = np.load('mm_matrix.npy')
            self.__mm_intercept = np.load('mm_intercept.npy')
        else:
            self.__count_matrix = self._plist_index()
            self.__mm_matrix, self.__mm_intercept = self._mmweight2matrix()

        self.__count_matrix = self._plist_index()

    def predict(self, query_peptide, loo=False):
        '''
        return the gps score for the query peptide
        :param query_peptide: (str) query peptide
        :param loo: (bool) if true, count_matrix will minus 1 according to the amino acid in each position in query peptide
        :return: gps score
        '''
        count_clone = self.__count_matrix * len(self.plist)
        matrix = np.zeros_like(self.__count_matrix)
        for i, a in enumerate(query_peptide):
            if a not in self.alist: a = 'C'
            if loo: count_clone[i, self.alist.index(a)] -= 1
            matrix[i, :] = self.__mm_matrix[self.alist.index(a), :]
        rm_num = 1 if loo else 0
        pls_count_matrix = (count_clone.T * self.pls_weight).T / (len(self.plist) - rm_num)
        return np.sum(matrix * pls_count_matrix) + self.__mm_intercept

    def generatePLSdata(self, query_peptide, loo=False):
        '''
        generate the pls vector of query peptide
        :param query_peptide: (str) query peptide
        :param loo: (bool) if true, the count_matrix will minus 1 according to the amino acid in each position in query peptide
        :return: (np.ndarray) the vector of feature for each position
        '''
        count_clone = self.__count_matrix * len(self.plist)
        matrix = np.zeros_like(count_clone)
        for i, a in enumerate(query_peptide):
            if a not in self.alist: a = 'C'
            if loo:
                count_clone[i, self.alist.index(a)] -= 1

            matrix[i, :] = self.__mm_matrix[self.alist.index(a), :]
        rm_num = 1 if loo else 0
        count_clone = (count_clone.T * self.pls_weight).T
        return np.sum(matrix * count_clone / (len(self.plist) - rm_num), 1)

    def generateMMdata(self, query_peptide, loo=False):
        count_clone = self.__count_matrix * len(self.plist)
        indicator_matrix = np.zeros_like(count_clone)
        for i, a in enumerate(query_peptide):
            if a not in self.alist: a = 'C'
            if loo: count_clone[i, self.alist.index(a)] -= 1
            indicator_matrix[i, self.alist.index(a)] = 1

        rm_num = 1 if loo else 0

        count_clone /= (len(self.plist) - rm_num)

        pls_count_matrix = (count_clone.T * self.pls_weight).T

        m = np.dot(indicator_matrix.T, pls_count_matrix) * self.__mm_matrix

        m += m.T

        np.fill_diagonal(m, np.diag(m) / float(2))

        iu1 = np.triu_indices(m.shape[0])

        return m[iu1]

    def getcutoff(self, randompeplist, sp=[0.98, 0.95, 0.85]):
        '''
        return cutoffs using 10000 random peptides as negative
        :param randompeplist: (list) random generated peptides
        :param sp: (float list) sp to be used for cutoff setting
        :return: (float list) cutoffs, same lens with sp
        '''
        rand_scores = sorted([self.predict(p) for p in randompeplist])
        cutoffs = np.zeros(len(sp))
        for i, s in enumerate(sp):
            index = np.floor(len(rand_scores) * s).astype(int)
            cutoffs[i] = rand_scores[index]
        return cutoffs

    def _plist_index(self):
        '''
        return the amino acid frequency on each position, row: position, column: self.alist, 61 x 24
        :return: count matrix
        '''
        n, m = len(self.plist[0]), len(self.alist)
        count_matrix = np.zeros((n, m))
        for i in range(n):
            for p in self.plist:
                count_matrix[i][self.alist.index(p[i])] += 1
        return count_matrix / float(len(self.plist))

    def _mmweight2matrix(self):
        '''
        convert matrix weight vector to similarity matrix, 24 x 24, index order is self.alist
        :return:
        '''
        aalist = self.getaalist()
        mm_matrix = np.zeros((len(self.alist), len(self.alist)))
        for n, d in enumerate(aalist):
            value = self.mm_weight[n + 1]
            i, j = self.alist.index(d[0]), self.alist.index(d[1])
            mm_matrix[i, j] = value
            mm_matrix[j, i] = value
        intercept = self.mm_weight[0]


        np.save('mm_matrix.npy', mm_matrix)
        np.save('mm_intercept.npy', intercept)

        return mm_matrix, intercept

    def getaalist(self):
        '''return aa-aa list
        AA: 0
        AR: 1
        '''
        aa = [self.alist[i] + self.alist[j] for i in range(len(self.alist)) for j in range(i, len(self.alist))]
        return aa

class AminoAcidDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError("embedding dimension should be divisible by number of heads")
        self.dropout = torch.nn.Dropout(0.1)
        self.projection_dim = embed_dim // num_heads
        self.query_dense = nn.Linear(embed_dim, embed_dim)
        self.key_dense = nn.Linear(embed_dim, embed_dim)
        self.value_dense = nn.Linear(embed_dim, embed_dim)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def attention(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)) / (key.size(-1) ** 0.5)
        weights = torch.nn.functional.softmax(scores, dim=-1)
        attention_weights = self.dropout(weights)
        output = torch.matmul(attention_weights, value)
        return output, attention_weights

    def separate_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.projection_dim)
        return x.transpose(1, 2)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        attention, weights = self.attention(query, key, value)
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.combine_heads(attention)
        return output, weights

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, inputs):
        attn_output, weights = self.attention(inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output), weights

class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = nn.Linear(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(maxlen, embed_dim)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0).expand_as(x[:, :, 0])
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class TransformerClassifier(nn.Module):
    def __init__(self, maxlen, vocab_size, embed_dim, num_heads, ff_dim, num_classes, num_layers=3):
        super(TransformerClassifier, self).__init__()
        self.embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)])
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(maxlen * embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding_layer(x)
        attention_weights = []
        for transformer_block in self.transformer_blocks:
            x, attn_weights = transformer_block(x)
            attention_weights.append(attn_weights)
        x = self.flatten(x)
        x = self.fc(x)
        return x, attention_weights

def seq2num(seq):
    out = []
    transdic = {'A': 8, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 0, 'L': 9, 'M': 10,
                'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, '*': 20}

    seq = seq.replace('U', '*').replace('X', '*').replace('O', '*')
    vec = [transdic[i] for i in seq]
    out.append(vec)
    out = np.array(out)
    return out

def num2onehot(num_array, vocab_size):
    onehot_array = np.eye(vocab_size)[num_array]
    return onehot_array

def transformer(seq_list):
    dict_feature = {}
    maxlen = 21
    vocab_size = 21
    embed_dim = 128
    num_heads = 4
    ff_dim = 64
    num_classes = 2


    model = TransformerClassifier(maxlen, vocab_size, embed_dim, num_heads, ff_dim, num_classes)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()


    for seq in seq_list:
        name = seq
        dict_seq2encod = {}
        dict_seq2weight = {}

        input_sequences_num = seq2num(seq)
        input_sequences_onehot = num2onehot(input_sequences_num, vocab_size)
        input_sequences_tensor = torch.tensor(input_sequences_onehot, dtype=torch.float32)


        with torch.no_grad():
            input_sequence = input_sequences_tensor.unsqueeze(0)
            outputs, attention_weights = model(input_sequence)
            embeddings = model.embedding_layer(input_sequence)
            dict_seq2encod[seq] = embeddings.squeeze(0).squeeze(0).mean(0).tolist()
            dict_seq2weight[seq] = [weight.squeeze(0).tolist() for weight in attention_weights]
            dict_feature[name] = dict_seq2encod[seq]

    return dict_feature

def seq2feature(seq_list):
    feature_all = []
    num = 0

    F1 = Protein_APAAC(seq_list)
    F2 = get_from_file(seq_list)
    F3 = Protein_CTDC(seq_list)
    F4 = Protein_DDE(seq_list)
    F5 = Protein_DistancePair(seq_list)
    F6 = Protein_EAAC(seq_list)
    F7 = ESM(seq_list)
    F8 = gps(seq_list)
    F9 = Protein_PAAC(seq_list)
    F10 = transformer(seq_list)

    for seq in seq_list:
        feature_list = [F1[seq],F2[seq],F3[seq],F4[seq],F5[seq],F6[seq],F7[seq],F8[seq],F9[seq],F10[seq]]
        feature_all.append(feature_list)

    return feature_all

def initialize_weights(model: nn.Module) -> None:
    """
    Initializes the weights of a model in place.

    :param model: An PyTorch model.
    """
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

class DNN(nn.Module):
    def __init__(self, char_size, embedding_size):
        super(DNN, self).__init__()
        self.char_size = char_size
        self.embedding_size = embedding_size

        self.first_linear_dim = self.char_size * self.embedding_size

        self.ffn_hidden_size = 100
        self.output_size = 1
        self.sigmoid = nn.Sigmoid()

        self.first = nn.Linear(self.first_linear_dim, self.ffn_hidden_size)
        self.hidden1 = nn.Linear(self.ffn_hidden_size, self.ffn_hidden_size)
        self.hidden2 = nn.Linear(self.ffn_hidden_size, self.ffn_hidden_size)
        self.hidden3 = nn.Linear(self.ffn_hidden_size, self.ffn_hidden_size)
        self.hidden4 = nn.Linear(self.ffn_hidden_size, self.ffn_hidden_size)
        self.hidden5 = nn.Linear(self.ffn_hidden_size, self.ffn_hidden_size)
        self.out = nn.Linear(self.ffn_hidden_size, self.output_size)

        initialize_weights(self)

    def forward(self, X):
        X = X.view(-1, self.char_size * self.embedding_size)

        output = F.relu(self.first(X))
        output = F.relu(self.hidden1(output))
        output = F.relu(self.hidden2(output))
        output = F.relu(self.hidden3(output))
        output = F.relu(self.hidden4(output))
        output = F.relu(self.hidden5(output))
        output = self.out(output)

        output = self.sigmoid(output)

        return output

def load_pretrain_model(path: str, dim):
    debug = info = print

    # Load model and args
    state = torch.load(path)
    loaded_state_dict = state['state_dict']

    # Build model
    model = DNN(dim, 1)
    model_state_dict = model.state_dict()

    # Skip missing parameters and parameters of mismatched size
    pretrained_state_dict = {}
    for param_name in loaded_state_dict.keys():


        if param_name not in model_state_dict:
            info(f'Warning: Pretrained parameter "{param_name}" cannot be found in model parameters.')
        elif model_state_dict[param_name].shape != loaded_state_dict[param_name].shape:
            info(f'Warning: Pretrained parameter "{param_name}" '
                 f'of shape {loaded_state_dict[param_name].shape} does not match corresponding '
                 f'model parameter of shape {model_state_dict[param_name].shape}.')
        else:
            # debug(f'Loading pretrained parameter "{param_name}".')
            pretrained_state_dict[param_name] = loaded_state_dict[param_name]

    # Load pretrained weights
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)

    return model

def test_predict(feature_path, model_path):
    feature_list = ['APAAC', 'CKSAAP', 'CTDC', 'DDE', 'DistancePair', 'EAAC', 'ESM', 'GPS','PAAC','transformer']
    Dim_list = [24, 1600, 39, 400, 20, 340, 1280, 231, 22, 128]
    score_list = []
    # num = 0
    for seq in range(len(feature_path)):
        feature_seq = feature_path[seq]
        pred = []
        for i in range(10):
            fn = feature_list[i]
            f = feature_seq[i]
            Dim = Dim_list[i]
            model = model_path +'model_'+ fn
            model_trained = load_pretrain_model(model, Dim)
            f = torch.Tensor(f)
            score = model_trained(f)
            pred.append(float(score.tolist()[0][0]))
            del model
            gc.collect()

        model_allname = model_path + 'model_all'
        model_all = load_pretrain_model(model_allname, 10)
        pred = torch.Tensor(pred)
        all_score = model_all(pred)
        score_list.append(all_score.tolist()[0][0])



    return score_list




if __name__ == '__main__':

    with open('independent_dataset_S.txt','r')as f:
        seq_list = []
        label_list = []
        readlines = f.readlines()
        for line in readlines:
            seq = line.rstrip().split('\t')[0]
            label = line.rstrip().split('\t')[1]
            seq_list.append(seq)
            label_list.append(label)




    feature_all = seq2feature(seq_list)





    score_list= test_predict(feature_all, model_path)
    with open('score.txt','w')as f:
        for i in range(len(seq_list)):
            f.write(seq_list[i] + '\t' + label_list[i] + '\t' + str(score_list[i]) + '\n')

