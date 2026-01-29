import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from Params import args
import scipy.sparse as sp
from Utils.TimeLogger import log
import torch as t
import torch.utils.data as data
import torch.utils.data as dataloader
import os
class DataHandler:
	def __init__(self):
		if args.data == 'yelp':
			predir = './Datasets/yelp/'
		elif args.data == 'lastfm':
			predir = './Datasets/lastfm/'
		elif args.data == 'beer':
			predir = './Datasets/beerAdvocate/'
		elif args.data == 'ciao':
			predir = './Datasets/ciao/'
		elif args.data == 'doubanbook':
			predir = './Datasets/doubanbook/'
		elif args.data == 'epinions':
			predir = './Datasets/epinions/'
		elif args.data == 'douban':
			predir = './Datasets/douban/'
		elif args.data == 'epinions':
			predir = './Datasets/epinions/'
		self.predir = predir

		if args.data == 'epinions':
			self.trnfile = predir + 'traindata.npy'  # 原：trnMat.pkl
			self.tstfile = predir + 'testdata.npy'  # 原：tstMat.pkl
			self.socialfile = predir + 'user_user_d.npy'  # 原：ufMat_1.pkl
		else:
			self.trnfile = predir + 'trnMat.pkl'
			self.tstfile = predir + 'tstMat.pkl'
			self.socialfile = predir + 'ufMat_1.pkl'
			self.rsocialfile = predir + 'ufsMat_dict.pkl'

		
		


	

	def loadOneFile(self, filename):     #加载数据集，并将数据集的矩阵格式转化为coo格式
		with open(filename, 'rb') as fs:
			ret = (pickle.load(fs) != 0).astype(np.float32)
		if type(ret) != coo_matrix:
			ret = sp.coo_matrix(ret)
		return ret
	
	

	# def loadOneFile(self, filename):     #加载数据集，并将数据集的矩阵格式转化为coo格式
	# 	file_ext = os.path.splitext(filename)[-1].lower()  # 获取文件后缀（.npy/.pkl）

	# 	# 1. 处理.pkl文件（对应非epinions数据集：yelp、lastfm等）
	# 	if file_ext == '.pkl':
	# 		with open(filename, 'rb') as fs:
	# 			ret = (pickle.load(fs) != 0).astype(np.float32)
	# 		if type(ret) != coo_matrix:
	# 			ret = sp.coo_matrix(ret)
	# 		return ret

	# 	# 2. 处理.npy文件（对应epinions数据集）
	# 	elif file_ext == '.npy':
	# 		ret = np.load(filename, allow_pickle=True)  

	# 		# 统一将数据转为list（方便后续处理）
	# 		if isinstance(ret, np.ndarray):
	# 			# 如果是一维numpy数组（如[int64]类型），转为list
	# 			if ret.ndim == 1:
	# 				interactions = ret.tolist()
	# 			else:
	# 				raise ValueError(f"不支持的numpy数组维度：{ret.ndim}，仅支持一维")
	# 		elif isinstance(ret, list):
	# 			interactions = ret
	# 		else:
	# 			raise ValueError(f"npy数据类型错误：{type(ret)}，仅支持list或一维numpy数组")

	# 		# 校验1：数据不能为空且长度为偶数（交替格式需u+i成对）
	# 		if len(interactions) == 0:
	# 			raise ValueError(f"{filename}为空！")
	# 		if len(interactions) % 2 != 0:
	# 			print(f"警告：数据长度{len(interactions)}为奇数，丢弃最后一个元素")
	# 			interactions = interactions[:-1]  # 丢弃最后一个，保证成对

	# 		# 校验2：所有元素必须是整数（用户/物品ID）
	# 		valid_interactions = []
	# 		for elem in interactions:
	# 			if isinstance(elem, (int, np.integer)):  # 支持python int和numpy int
	# 				valid_interactions.append(int(elem))  # 统一转为int
	# 			else:
	# 				print(f"跳过非整数元素：{elem}（类型：{type(elem)}）")
	# 		if len(valid_interactions) < 2:
	# 			raise ValueError(f"{filename}中有效整数元素不足2个，无法组成交互对")

	# 		# 核心：解析一维交替数组为（u,i）交互对
	# 		all_users = []
	# 		all_items = []
	# 		# 步长2遍历：0→u,1→i；2→u,3→i...
	# 		for idx in range(0, len(valid_interactions), 2):
	# 			u = valid_interactions[idx]
	# 			i = valid_interactions[idx + 1]
	# 			all_users.append(u)
	# 			all_items.append(i)

	# 		# 构建用户-物品二维矩阵
	# 		user_count = max(all_users) + 1
	# 		item_count = max(all_items) + 1
	# 		ret_matrix = np.zeros((user_count, item_count), dtype=np.float32)
	# 		for u, i in zip(all_users, all_items):
	# 			ret_matrix[u, i] = 1.0  # 标记交互

	# 		# 转为coo稀疏矩阵（兼容后续逻辑）
	# 		ret_matrix = (ret_matrix != 0).astype(np.float32)
	# 		if type(ret_matrix) != coo_matrix:
	# 			ret_matrix = sp.coo_matrix(ret_matrix)
	# 		return ret_matrix
    
	# 	# 3. 未知格式报错
	# 	else:
	# 		raise ValueError(f"不支持的文件格式：{file_ext}，仅支持.npy和.pkl")
	
	def normalizeAdj(self, mat):  #归一化邻接矩阵
		degree = np.array(mat.sum(axis=-1))
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0
		dInvSqrtMat = sp.diags(dInvSqrt) #将处理过的度数组转换为对角矩阵
		return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

	def makeTorchAdj(self, mat):
		#创建用户-物品邻接矩阵，并将此矩阵归一化，并转化为pytorch适合处理的稀疏张量的形式
		#这个邻接矩阵左上角是用户-用户关系矩阵，右上角是用户-物品关系矩阵，左下角是物品-用户关系矩阵，右下角是物品-物品关系矩阵
		#在这个矩阵中，a关系矩阵和b关系矩阵是空的
		# make ui adj
		a = sp.csr_matrix((args.user, args.user))
		b = sp.csr_matrix((args.item, args.item))
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		mat = (mat != 0) * 1.0
		mat = (mat + sp.eye(mat.shape[0])) * 1.0
		mat = self.normalizeAdj(mat)  #归一化过程，A=DAD


		# make cuda tensor
		idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = t.from_numpy(mat.data.astype(np.float32))
		shape = t.Size(mat.shape)
		return t.sparse.FloatTensor(idxs, vals, shape).cuda()

	def makesocialAdj(self, mat):
		mat = self.normalizeAdj(mat)   #归一化操作
		# print(mat)
		idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = t.from_numpy(mat.data.astype(np.float32))
		shape = t.Size(mat.shape)
		return t.sparse.FloatTensor(idxs, vals, shape).cuda()

	def LoadData(self):  #加载测试集和训练集,加载用户社交网络
		trnMat = self.loadOneFile(self.trnfile)
		tstMat = self.loadOneFile(self.tstfile)
		ufMat = self.loadOneFile(self.socialfile)
		self.trnMat = trnMat
		args.user, args.item = trnMat.shape
		self.torchBiAdj = self.makeTorchAdj(trnMat)

		self.usmat=ufMat
		args.user_social, args.user_social = ufMat.shape
		self.torchSocialAdj = self.makesocialAdj(ufMat)

		trnData = TrnData(trnMat, ufMat)
		self.trnLoader = dataloader.DataLoader(trnData, batch_size=args.batch, shuffle=True, num_workers=0)
		tstData = TstData(tstMat, trnMat)
		self.tstLoader = dataloader.DataLoader(tstData, batch_size=args.tstBat, shuffle=False, num_workers=0)

class TrnData(data.Dataset):
	def __init__(self, coomat, uucoomat):
		self.rows = coomat.row
		self.cols = coomat.col
		self.dokmat = coomat.todok()
		self.uurows = uucoomat.row
		self.uuDokmat = uucoomat.todok()
		self.negs = np.zeros(len(self.rows)).astype(np.int32)
		self.uuNegs = np.zeros(len(self.uurows)).astype(np.int32)
		 # 扩展负样本存储：支持混合采样（1硬+1随机，故长度为2*num_interactions）
        # self.negs = np.zeros(len(self.rows) * 2).astype(np.int32)  # 总负样本列表
        # self.uuNegs = np.zeros(len(self.uurows)).astype(np.int32)  # 社交负样本（原有逻辑保留）

	# def negSampling(self, item_emb=None, hard_ratio=0.5):
    #     # """
    #     # 混合负采样：硬负采样 + 随机负采样
    #     # Args:
    #     #     item_emb: 模型实时物品嵌入 [item_num, emb_dim]（GPU张量）
    #     #     hard_ratio: 硬负样本占比（0~1，此处固定0.5，即1硬+1随机）
    #     # """
    #     num_total_negs = len(self.negs)  # 总负样本数 = 2 * 正样本数
    #     num_hard_negs = int(num_total_negs * hard_ratio)  # 硬负样本数
    #     num_rand_negs = num_total_negs - num_hard_negs  # 随机负样本数
    #     # -------------------------- 1. 硬负采样（基于物品嵌入相似度）--------------------------
    # 	if item_emb is not None and num_hard_negs > 0:
	# 		# 1.1 物品嵌入转CPU（避免GPU内存占用，相似度计算在CPU）
	# 		item_emb_cpu = item_emb.detach().cpu()  # [item_num, emb_dim]
	# 		item_num = item_emb_cpu.shape[0]  # 总物品数

	# 		# 1.2 批量获取所有正样本物品的嵌入（避免循环取数，提升效率）
	# 		pos_item_emb = item_emb_cpu[self.cols]  # [num_interactions, emb_dim]

	# 		# 1.3 计算正样本与所有物品的相似度（点积相似度，等价于余弦相似度（若嵌入已归一化））
	# 		# 结果形状：[num_interactions, item_num]
	# 		item_sim = torch.matmul(pos_item_emb, item_emb_cpu.T)

    #         # 1.4 过滤掉「已交互的正样本」（相似度设为-∞，避免采样到正样本）
    #         # 遍历每个正样本，标记其交互过的物品
    #         for idx in range(len(self.rows)):
    #             u = self.rows[idx]
    #             # 获取用户u的所有正样本物品ID（从dokmat中筛选）
    #             pos_items_for_u = [i for (uu, i) in self.dokmat.keys() if uu == u]
    #             # 将这些正样本的相似度设为-∞，排除采样可能
    #             item_sim[idx, pos_items_for_u] = -float('inf')

    #         # 1.5 对每个正样本，选择相似度最高的物品作为硬负样本
    #         # topk(1)：取相似度最高的1个；indices：获取其物品ID
    #         hard_neg_indices = torch.topk(item_sim, k=1, dim=1).indices.squeeze()  # [num_interactions]
    #         # 确保硬负样本ID在有效范围内（防止嵌入维度与实际物品数不匹配）
    #         hard_neg_indices = torch.clamp(hard_neg_indices, 0, item_num - 1).numpy()

    #         # 1.6 填充硬负样本到总负样本列表（前num_hard_negs个位置）
    #         # 注：num_hard_negs = num_interactions（因hard_ratio=0.5，总负样本数=2*num_interactions）
    #         self.negs[:num_hard_negs] = hard_neg_indices

    #     # -------------------------- 2. 随机负采样（保障多样性）--------------------------
    #     # 采样位置：总负样本列表的后num_rand_negs个位置
    #     rand_neg_start = num_hard_negs
	# 	for i in range(num_rand_negs):
    #         # 对应正样本的索引（每2个负样本对应1个正样本）
    #         pos_idx = i % len(self.rows)
    #         u = self.rows[pos_idx]
    #         # 随机采样，直到找到非交互物品
    #         while True:
    #             iNeg = np.random.randint(args.item)  # args.item：总物品数（从配置读取）
    #             if (u, iNeg) not in self.dokmat:
    #                 break
    #         self.negs[rand_neg_start + i] = iNeg

    #     # -------------------------- 3. 社交负样本（原有随机逻辑保留）--------------------------
    #     for i in range(len(self.uurows)):
    #         u = self.uurows[i]
    #         while True:
    #             uNeg = np.random.randint(args.user)
    #             if (u, uNeg) not in self.uuDokmat:
    #                 break
    #         self.uuNegs[i] = uNeg
	
	def negSampling(self):    #此处生成了负样本用于用户的训练
		for i in range(len(self.rows)):
			u = self.rows[i]
			while True:
				iNeg = np.random.randint(args.item)
				if (u, iNeg) not in self.dokmat:
					break
			self.negs[i] = iNeg
		for i in range(len(self.uurows)):
			u = self.uurows[i]
			while True:
				uNeg = np.random.randint(args.user)
				if (u, uNeg) not in self.uuDokmat:
					break
			self.uuNegs[i] = uNeg


	def __len__(self):
		return len(self.rows)

	def __getitem__(self, idx):
		return self.rows[idx], self.cols[idx], self.negs[idx]

class TstData(data.Dataset):
	def __init__(self, coomat, trnMat):
		self.csrmat = (trnMat.tocsr() != 0) * 1.0

		tstLocs = [None] * coomat.shape[0]
		tstUsrs = set()
		for i in range(len(coomat.data)):
			row = coomat.row[i]
			col = coomat.col[i]
			if tstLocs[row] is None:
				tstLocs[row] = list()
			tstLocs[row].append(col)
			tstUsrs.add(row)
		tstUsrs = np.array(list(tstUsrs))
		self.tstUsrs = tstUsrs
		self.tstLocs = tstLocs

	def __len__(self):
		return len(self.tstUsrs)

	def __getitem__(self, idx):
		return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])