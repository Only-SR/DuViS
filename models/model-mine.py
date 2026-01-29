import torch
from Utils.TimeLogger import log
from Params import args
from Model import Model, vgae_encoder, vgae_decoder, vgae, SocialDenoising, SDNet
from DataHandler import DataHandler
import numpy as np
from Utils.Utils import calcRegLoss, pairPredict, l2_reg_loss, Metric
from copy import deepcopy
import logging
import sys
import os,time
from Diffusion_process import DiffusionProcess
from Louvain import Louvain
from Louvain1 import Louvain1
import networkx as nx
import community as community_louvain
from collections import defaultdict
from GatedFusion import GatedFusion
from MemoryFusion import MemoryFusion
from torch import nn
start_time = time.time()
import igraph as ig
from leidenalg import find_partition, ModularityVertexPartition


class Coach:
    def __init__(self, handler):
        self.item_emb_trained = None
        self.user_emb_trained = None
        self.handler = handler
        self.device = torch.device('cuda:'+args.device if torch.cuda.is_available() else 'cpu')
        print('USER', args.user, 'ITEM', args.item)
        print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

    def makePrint(self, name, ep, reses, save): 
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    def run(self):
        self.prepareModel()
        log('Model Prepared')

        recall10Max = 0
        ndcg10Max = 0
        recall20Max = 0
        ndcg20Max = 0
        recall40Max = 0
        ndcg40Max = 0
        bestEpoch10 = 0
        bestEpoch20 = 0
        bestEpoch40 = 0
        stloc = 0
        patience = 0
        log('Model Initialized')
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
        log_save = './History/' + args.data + '/'
        log_file = args.save_name
        fname = f'{log_file}.txt'
        fh = logging.FileHandler(os.path.join(log_save, fname))
        fh.setFormatter(logging.Formatter(log_format))
        logger = logging.getLogger()
        logger.addHandler(fh)
        logger.info(args)
        logger.info('================')
        for ep in range(stloc, args.epoch):
            #temperature = max(0.05, args.init_temperature * pow(args.temperature_decay, ep))
            temperature = 0.3
            tstFlag = (ep % args.tstEpoch == 0)
            reses = self.trainEpoch(temperature,ep)
            log(self.makePrint('Train', ep, reses, tstFlag))
            if patience > 200:
                break 
            if tstFlag:
                reses = self.testEpoch()
                if (recall10Max > reses['Recall10']):
                    patience = patience +1
                if reses['Recall10'] > recall10Max:
                    recall10Max = reses['Recall10']
                    ndcg10Max = reses['NDCG10']
                    bestEpoch10 = ep
                # log(self.makePrint('Test', ep, reses, tstFlag))
                if reses['Recall20'] > recall20Max:
                    recall20Max = reses['Recall20']
                    ndcg20Max = reses['NDCG20']
                    bestEpoch20 = ep
                # log(self.makePrint('Test', ep, reses, tstFlag))
                if reses['Recall40'] > recall40Max:
                    recall40Max = reses['Recall40']
                    ndcg40Max = reses['NDCG40']
                    bestEpoch40 = ep
                log(self.makePrint('Test', ep, reses, tstFlag))

            print('recall', recall10Max, recall20Max, recall40Max)
            print('ndcg', ndcg10Max, ndcg20Max, ndcg40Max)
            print()
            
            logger.info('+ epoch {} tested, elapsed {:.2f}s, Recall@{}: {:.4f}, NDCG@{}: {:.4f}'.format(
            ep, time.time() - start_time, 10,  reses['Recall10'], 10, reses['NDCG10']))
            logger.info('+ epoch {} tested, elapsed {:.2f}s, Recall@{}: {:.4f}, NDCG@{}: {:.4f}'.format(
            ep, time.time() - start_time, 20,  reses['Recall20'], 20, reses['NDCG20']))
            logger.info('+ epoch {} tested, elapsed {:.2f}s, Recall@{}: {:.4f}, NDCG@{}: {:.4f}'.format(
            ep, time.time() - start_time, 40,  reses['Recall40'], 40, reses['NDCG40']))
        print('Best epoch : ', bestEpoch10, ' , Recall : ', recall10Max, ' , NDCG : ', ndcg10Max)
        print('Best epoch : ', bestEpoch20, ' , Recall : ', recall20Max, ' , NDCG : ', ndcg20Max)
        print('Best epoch : ', bestEpoch40, ' , Recall : ', recall40Max, ' , NDCG : ', ndcg40Max)
        logger.info('+ epoch {} tested, elapsed {:.2f}s, Recall@{}: {:.4f}, NDCG@{}: {:.4f}'.format(
        ep, time.time() - start_time, 10,  recall10Max, 10, ndcg10Max))
        logger.info('+ epoch {} tested, elapsed {:.2f}s, Recall@{}: {:.4f}, NDCG@{}: {:.4f}'.format(
        ep, time.time() - start_time, 20,  recall20Max, 20, ndcg20Max))
        logger.info('+ epoch {} tested, elapsed {:.2f}s, Recall@{}: {:.4f}, NDCG@{}: {:.4f}'.format(
        ep, time.time() - start_time, 40,  recall40Max, 40, ndcg40Max))
    def prepareModel(self):
        self.model = Model().cuda()
        encoder = vgae_encoder().cuda()
        decoder = vgae_decoder().cuda()
        self.generator_1 = vgae(encoder, decoder).cuda()
         # 对社交邻接矩阵进行合并操作，解决uncoalesced tensor问题
        social_adj = self.handler.torchSocialAdj.coalesce()
        self.social_denosing = SocialDenoising(social_adj)
        data = deepcopy(self.handler.torchBiAdj.coalesce())

        self.new_socialadj = deepcopy(self.social_denosing.build_socialgraph(self.model.uEmbeds.weight))
        self.new_UU, self.new_UI = self.social_denosing.denoise_graph(social_adj,data,self.model.uEmbeds.weight,self.model.iEmbeds.weight)
        self.social_pairs,_ = self.social_denosing.get_all_social_pairs(self.new_socialadj)#原始社交图社交对
        self.H,self.comunity =self.build_community_hypergraph(self.social_pairs) #社区超图

        self.louvain = Louvain().cuda()
        hyper_hidden_dims = [64, 128, 64]  # 超图分支隐藏层
        gcn_hidden_dims = [128, 256, 128]    # GCN分支隐藏层
        self.louvain1 = Louvain1(in_dim=args.latdim,hyper_hidden_dims=hyper_hidden_dims,gcn_hidden_dims=gcn_hidden_dims,num_heads=4,dropout=0.2).cuda()
        #加入扩散
        output_dims = [args.dims] + [args.latdim]
        input_dims = output_dims[::-1]
        N=args.user
        self.SDNet1 = SDNet([N,2*args.latdim], [2*args.latdim,N], args.emb_size, time_type="cat", norm=args.norm).to(self.device)
        self.SDNet2 = SDNet(input_dims, output_dims, args.emb_size, time_type="cat", norm=args.norm).to(self.device)
        self.DiffProcess1=DiffusionProcess(args.noise_schedule,args.noise_scale, args.noise_min, args.noise_max, args.steps,self.device).to(self.device)
        self.DiffProcess2=DiffusionProcess(args.noise_schedule,args.noise_scale, args.noise_min, args.noise_max, args.steps,self.device).to(self.device)
        self.optimizer1 = torch.optim.Adam([
             {'params':  self.SDNet1.parameters(), 'weight_decay': 0},
         ], lr=args.difflr)
        # self.optimizer2 = torch.optim.Adam([
        #      {'params':  self.SDNet2.parameters(), 'weight_decay': 0},
        #  ], lr=args.difflr)
        

        self.opt_gen_1 = torch.optim.Adam(self.generator_1.parameters(), lr=args.lr)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.opt_d = torch.optim.Adam(self.social_denosing.parameters(), lr=args.lr)
        self.opt_l = torch.optim.Adam(self.louvain1.parameters(), lr=args.lr)#社区检测算法
        
        #self.opt_g = torch.optim.Adam(self.gate.parameters(), lr=args.lr)#门控去噪

        #self.opt_g = torch.optim.Adam(self.gatedfusion.parameters(), lr=args.lr)#门控融合
        #self.opt_m = torch.optim.Adam(self.memoryfusion.parameters(), lr=args.lr)#门控融合

    def trainEpoch(self, temperature,ep):
        trnLoader = self.handler.trnLoader
        trnLoader.dataset.negSampling()
        generate_loss_1, bpr_loss, im_loss, ib_loss, reg_loss = 0, 0, 0, 0, 0
        steps = trnLoader.dataset.__len__() // args.batch
        data = deepcopy(self.handler.torchBiAdj).cuda()
        data2 = self.generator_generate(self.generator_1)
        socialadj = deepcopy(self.handler.torchSocialAdj)
        init_user = self.model.uEmbeds.weight#模型初始用户嵌入
        
        # if ep==0 :
        #     self.social_pairs, _ = self.social_denosing.get_all_social_pairs(self.new_UU)
        #     self.H, self.comunity = self.build_community_hypergraph(self.social_pairs)  # 重新构建社区
        # if ep % 5 == 0 & ep!=0 :  # ep是当前epoch，需传入trainEpoch
        #     self.social_pairs, _ = self.social_denosing.get_all_social_pairs(self.new_UU)
        #     self.H, self.comunity = self.build_community_hypergraph(self.social_pairs)  # 重新构建社区
            #self.new_socialadj = self.social_denosing.build_socialgraph(self.model.social_uu)
        if ep==0 :
            self.new_socialadj = self.social_denosing.build_socialgraph(init_user)
            #self.new_socialadj
            self.social_pairs, _ = self.social_denosing.get_all_social_pairs(init_user)
            self.H, self.comunity = self.build_community_hypergraph(self.social_pairs)  # 重新构建社区
        if ep % args.interval == 0 & ep!=0 :  # ep是当前epoch，需传入trainEpoch
            self.new_socialadj = self.social_denosing.build_socialgraph(init_user)
            self.social_pairs, _ = self.social_denosing.get_all_social_pairs(self.new_socialadj)
            self.H, self.comunity = self.build_community_hypergraph(self.social_pairs)  # 重新构建社区
        # self.social_pairs,_ = self.social_denosing.get_all_social_pairs(new_socialadj)#去噪社交图社交对
        # self.H,self.comunity =self.build_community_hypergraph(self.social_pairs) #社区超图
        self.new_UU, self.new_UI = self.social_denosing.denoise_graph(socialadj.coalesce(),data.coalesce(),self.model.uEmbeds.weight,self.model.iEmbeds.weight)
        #self.new_socialadj.coalesce() or socialadj.coalesce()
        for i, tem in enumerate(trnLoader):            
            self.opt.zero_grad()
            self.opt_gen_1.zero_grad()
            self.opt_d.zero_grad()
            self.opt_l.zero_grad()
            self.optimizer1.zero_grad()
            #self.optimizer1.zero_grad()
            # self.optimizer2.zero_grad()
            ancs, poss, negs = tem 
            ancs = ancs.long().cuda()
            poss = poss.long().cuda()
            negs = negs.long().cuda()
            #对社交图用户嵌入进行扩散
            # self.socialadj_diff,weight1=self.DiffProcess1.get_output(self.SDNet1, socialadj, args.reweight)
            # mse1 = self.mean_flat((self.socialadj_diff - socialadj) ** 2)
            # diff_loss1=(weight1 * mse1).mean() 
            ui1,uu1 = self.model.forward_fusion(self.new_UI, self.new_socialadj)
            ui2,uu2 = self.model.forward_fusion(data2, self.new_socialadj)


            #融合用户嵌入
            inter_u1 = ui1[:args.user]#UI图用户嵌入
            m0 = self.model.MLP0(uu1, inter_u1)#两用户嵌入融合
            m01 = torch.reshape(m0[:, 0], [args.user, 1])
            m02 = torch.reshape(m0[:, 1], [args.user, 1])
            user1=uu1 * m01 + inter_u1 * m02
            #user1=uu1+inter_u1
            item1=ui1[args.user:]
            

            #对社交图用户嵌入进行扩散
            # pre_user2 = uu2
            # user_d2,weight2=self.DiffProcess2.get_output(self.SDNet2, pre_user2, args.reweight)
            # mse2 = self.mean_flat((user_d2 - pre_user2) ** 2)
            # diff_loss2=(weight2 * mse2).mean() 
            #融合用户嵌入
            inter_u2 = ui2[:args.user]#UI图用户嵌入
            m1 = self.model.MLP1(uu2, inter_u2)#两用户嵌入融合
            m11 = torch.reshape(m1[:, 0], [args.user, 1])
            m12 = torch.reshape(m1[:, 1], [args.user, 1])
            user2=uu2 * m11 + inter_u2 * m12
            #user2=uu2+inter_u2
            item2=ui2[args.user:]
           
            out1 = torch.cat((user1 , item1), dim=0)
            out2 = torch.cat((user2 , item2), dim=0)
            #self.new_socialadj = self.social_denosing.build_socialgraph(self.model.social_uu)  # 最新社交图
            #user_enhanced = self.louvain(init_user, self.social_pairs, self.H, self.comunity)#还可以用gcn后得用户嵌入
            
            # user1 = out1_[:args.user]
            # item1 = out1_[args.user:]
            # user2 = out2_[:args.user]
            # item2 = out2_[args.user:]
            #user = self.gate(user2,user1)
            #userEmbed= user1+user2

            #user_enhanced = self.louvain1(init_user,self.H)

            #user_enhanced1 = self.louvain(self.model.social_uu, self.social_pairs, self.H, self.comunity)#还可以用gcn后得用户嵌入
           

            #直接相加融合嵌入
            #user=args.a*user_enhanced+args.b*(user1+user2)

            # m3 = self.model.MLP3(item1, item2)#两用户嵌入融合
            # m31 = torch.reshape(m3[:, 0], [args.item, 1])
            # m32 = torch.reshape(m3[:, 1], [args.item, 1])
            user=user1+user2
            item= item1+item2

            ancEmbeds = user[ancs]
            posEmbeds = item[poss]
            negEmbeds = item[negs]

            pos_score = torch.mul(ancEmbeds, posEmbeds).sum(dim=1)
            neg_score = torch.mul(ancEmbeds, negEmbeds).sum(dim=1)
            loss_b = -torch.log(10e-6 + torch.sigmoid(pos_score - neg_score))
            bprLoss = torch.mean(loss_b)

            regLoss = calcRegLoss(self.model) * args.lambda3
            loss = bprLoss + regLoss 
            bpr_loss += float(bprLoss)
            reg_loss += float(regLoss) 

            loss_1 = self.generator_1(deepcopy(self.handler.torchBiAdj).cuda(), ancs, poss, negs)
            generate_loss_1 += float(loss_1)

            #loss_cl = self.model.loss_graphcl(uu1,inter_u1, ancs).mean() * args.ssl_reg

            loss_user, loss_item = self.model.loss_graphcl_2(out1, out2, ancs, poss)
            loss_cl = loss_user.mean()+loss_item.mean() 
            loss=loss+args.lambda2*loss_1+args.lambda1*loss_cl

            im_loss += float(loss)  
            loss.backward()
            self.opt.step()
            self.opt_gen_1.step()
            self.opt_d.step()
            self.opt_l.step()
            #self.optimizer1.step()
            # self.optimizer2.step()
            # with torch.no_grad():
            #     self.user_emb_trained, self.item_emb_trained = self.model.forward_gcn(self.handler.torchBiAdj)
            with torch.no_grad():
                #socialadj_d = self.DiffProcess1.p_sample(self.SDNet1, socialadj, args.sampling_steps, args.sampling_noise)
                ui1,uu1 = self.model.forward_fusion(self.new_UI, self.new_socialadj)
                ui2,uu2 = self.model.forward_fusion(data2, self.new_socialadj)
                # ui1,uu1 = self.model.forward_fusion(data, self.new_socialadj)
                # ui2,uu2 = self.model.forward_fusion(data2, self.new_socialadj)
                #对社交图用户嵌入进行扩散
                #user_d1 = self.DiffProcess1.p_sample(self.SDNet1, uu1, args.sampling_steps, args.sampling_noise)
                #融合用户嵌入
                inter_u1 = ui1[:args.user]#UI图用户嵌入
                m0 = self.model.MLP0(uu1, inter_u1)#两用户嵌入融合
                m01 = torch.reshape(m0[:, 0], [args.user, 1])
                m02 = torch.reshape(m0[:, 1], [args.user, 1])
                user1=uu1 * m01 + inter_u1 * m02
                #user1=user_d1+inter_u1
                item1=ui1[args.user:]

                #user_d2 = self.DiffProcess2.p_sample(self.SDNet2, uu2, args.sampling_steps, args.sampling_noise)
                #融合用户嵌入
                inter_u2 = ui2[:args.user]#UI图用户嵌入
                m1 = self.model.MLP1(uu2, inter_u2)#两用户嵌入融合
                m11 = torch.reshape(m1[:, 0], [args.user, 1])
                m12 = torch.reshape(m1[:, 1], [args.user, 1])
                user2=uu2 * m11 + inter_u2 * m12
                #user1=user_d1+inter_u1
               
                item2=ui2[args.user:]


                # out1_ = self.model.forward_fusion(data, self.new_socialadj)
                # out2_ = self.model.forward_fusion(data2, self.new_socialadj)    
                # user_enhanced = self.louvain(init_user,self.social_pairs,self.H,self.comunity)
                

                #user_enhanced = self.louvain(self.model.social_uu,self.social_pairs,self.H,self.comunity)

    
                # m2 = self.model.MLP2(user1, user2)#两用户嵌入融合
                # m21 = torch.reshape(m2[:, 0], [args.user, 1])
                # m22 = torch.reshape(m2[:, 1], [args.user, 1])
                # user=m21*user1+m22*user2
                usrEmbeds = user1+user2
                # m3 = self.model.MLP3(item1, item2)#两用户嵌入融合
                # m31 = torch.reshape(m3[:, 0], [args.item, 1])
                # m32 = torch.reshape(m3[:, 1], [args.item, 1])

                # item= m31*item1+m32*item2
                itmEmbeds = item1+item2
                
                self.user_emb_trained = usrEmbeds

                self.item_emb_trained = itmEmbeds

            log('Step %d/%d: gen 1 : %.3f ; bpr : %.3f ; im : %.3f ; ib : %.3f ; reg : %.3f  ' % (
                i,
                steps,
                generate_loss_1,
                bpr_loss,
                im_loss,
                ib_loss,
                reg_loss,
            ), save=False, oneline=True)

        ret = dict()
        ret['Gen_1 Loss'] = generate_loss_1 / steps
        ret['BPR Loss'] = bpr_loss / steps
        ret['IM Loss'] = im_loss / steps
        ret['IB Loss'] = ib_loss / steps
        ret['Reg Loss'] = reg_loss / steps
        ret['loss'] = bpr_loss / steps + im_loss / steps + reg_loss / steps + ib_loss / steps

        return ret
    def mean_flat(self,tensor):
        """
        Take the mean over all non-batch dimensions.
        """
        return tensor.mean(dim=list(range(1, len(tensor.shape))))
    
    # def build_community_hypergraph(self, social_relations): #构建超图
    #         """使用Louvain检测社区并构建超图"""
    #         # 构建社交图
    #         G = nx.Graph()
    #         G.add_edges_from(social_relations)
    #         # Louvain社区检测
    #         partition = community_louvain.best_partition(G)
            
    #         # 构建社区到用户的映射
    #         communities = defaultdict(list)
    #         for user, comm_id in partition.items():
    #             communities[comm_id].append(user)
            
    #         # 构建超图关联矩阵 H: [num_users, num_communities]
    #         num_communities = len(communities)
    #         H = torch.zeros(args.user, num_communities)
            
    #         for comm_id, users in communities.items():
    #             for user in users:
    #                 if user < args.user:  # 确保用户ID在范围内
    #                     H[user, comm_id] = 1.0
            
    #         return H, communities

    def build_community_hypergraph(self, social_relations): #构建超图
            """使用Louvain检测社区并构建超图"""
            # 构建社交图
            G = nx.Graph()
            G.add_edges_from(social_relations)
            # Louvain社区检测
            # 转换为igraph
            G_ig = ig.Graph.from_networkx(G)
            
            # 运行Leiden
            partition = find_partition(
                G_ig,
                ModularityVertexPartition,
                seed=42
            )
            
            # 构建社区到用户的映射
            communities = defaultdict(list)
            for user, comm_id in enumerate(partition.membership):
                communities[comm_id].append(user)
            
            # 构建超图关联矩阵 H: [num_users, num_communities]
            num_communities = len(communities)
            H = torch.zeros(args.user, num_communities)
            
            for comm_id, users in communities.items():
                for user in users:
                    if user < args.user:  # 确保用户ID在范围内
                        H[user, comm_id] = 1.0
            
            return H, communities

    def testEpoch(self):
        tstLoader = self.handler.tstLoader
        epRecall10, epNdcg10 = [0] * 2
        epRecall20, epNdcg20 = [0] * 2
        epRecall40, epNdcg40 = [0] * 2
        i = 0
        num = tstLoader.dataset.__len__()
        steps = num // args.tstBat
        for usr, trnMask in tstLoader:
            i += 1
            usr = usr.long().cuda()
            trnMask = trnMask.cuda()
            # usrEmbeds, itmEmbeds = self.model.forward_gcn(self.handler.torchBiAdj)
            usrEmbeds = self.user_emb_trained
            itmEmbeds = self.item_emb_trained
            allPreds = torch.mm(usrEmbeds[usr], torch.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 10e8
            _, topLocs10 = torch.topk(allPreds, args.topk10)
            recall10, ndcg10 = self.calcRes(topLocs10.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr,
                                            args.topk10)
            _, topLocs20 = torch.topk(allPreds, args.topk20)
            recall20, ndcg20 = self.calcRes(topLocs20.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr,
                                            args.topk20)
            _, topLocs40 = torch.topk(allPreds, args.topk40)
            recall40, ndcg40 = self.calcRes(topLocs40.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr,
                                            args.topk40)
            epRecall10 += recall10
            epNdcg10 += ndcg10
            epRecall20 += recall20
            epNdcg20 += ndcg20
            epRecall40 += recall40
            epNdcg40 += ndcg40
            log('Steps %d/%d: recall10 = %.2f, ndcg10 = %.2f ,recall20 = %.2f, ndcg20 = %.2f ,recall40 = %.2f, ndcg40 = %.2f         '
                % (i, steps, recall10, ndcg10, recall20, ndcg20, recall40, ndcg40, ), save=False,
                oneline=True)
        ret = dict()
        ret['Recall10'] = epRecall10 / num
        ret['NDCG10'] = epNdcg10 / num
        ret['Recall20'] = epRecall20 / num
        ret['NDCG20'] = epNdcg20 / num
        ret['Recall40'] = epRecall40 / num
        ret['NDCG40'] = epNdcg40 / num
        return ret

    def calcRes(self, topLocs, tstLocs, batIds, topk):
        assert topLocs.shape[0] == len(batIds)
        allRecall = allNdcg = 0
        for i in range(len(batIds)):
            temTopLocs = list(topLocs[i])
            temTstLocs = tstLocs[batIds[i]]
            tstNum = len(temTstLocs)
            maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, topk))])
            recall = dcg = 0
            for val in temTstLocs:
                if val in temTopLocs:
                    recall += 1
                    dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
            recall = recall / tstNum
            ndcg = dcg / maxDcg
            allRecall += recall
            allNdcg += ndcg
        return allRecall, allNdcg

    def generator_generate(self, generator):
        edge_index = []
        edge_index.append([])
        edge_index.append([])
        adj = deepcopy(self.handler.torchBiAdj)
        idxs = adj._indices()

        with torch.no_grad():
            view = generator.generate(self.handler.torchBiAdj, idxs, adj)

        return view
    
class inter_denoise_gate(torch.nn.Module):
    def __init__(self):
        super(inter_denoise_gate, self).__init__()
        # 定义3个线性层，分别处理CF嵌入、社交嵌入、两者交互
        self.W_cf = nn.Linear(args.latdim, args.latdim)  # 处理CF域嵌入
        self.W_soc = nn.Linear(args.latdim, args.latdim)  # 处理社交域嵌入
        self.W_mix = nn.Linear(args.latdim, args.latdim)  # 处理CF与社交的交互（element-wise乘）
        
        # 线性层：融合CF与社交的拼接特征（用于增强门控）
        self.W_en = nn.Linear(int(2*args.latdim), args.latdim)
        
        self.sig = nn.Sigmoid()  # 门控权重（输出0~1，控制信息保留比例）
        self.tanh = nn.Tanh()    # 激活函数（压缩嵌入范围，增强非线性）
        
    def forward(self, embs_cf, embs_soc):
        # embs_cf：CF域嵌入（shape: [num_users/items, dim]）
        # embs_soc：社交域嵌入（shape: [num_users, dim]）
        
        # 1. 计算「遗忘门（forget gate）」：筛选社交嵌入中与CF嵌入一致的有效信息
        # W_cf(embs_cf) + W_soc(embs_soc) + W_mix(embs_soc * embs_cf) → 融合特征
        # sigmoid → 遗忘权重（0：完全遗忘社交信息，1：完全保留社交信息）
        embs_mix = self.W_cf(embs_cf) + self.W_soc(embs_soc) + self.W_mix(embs_soc * embs_cf)
        forget_gate = self.sig(embs_mix)
        
        # 2. 计算「增强门（enhance gate）」：强化社交嵌入中有用的局部特征
        # 拼接CF与社交嵌入 → W_en线性变换 → sigmoid → 增强权重
        embs_concat = self.W_en(torch.cat((embs_cf, embs_soc), dim=-1))  # dim=-1：在嵌入维度拼接
        enhance_gate = self.sig(embs_concat)
        
        # 3. 输出去噪后的社交嵌入：遗忘噪声 + 增强有效信息
        # forget_gate * embs_soc → 保留有效社交信息
        # enhance_gate * tanh(W_soc(embs_soc)) → 增强社交嵌入的非线性表达
        out = forget_gate * embs_soc + enhance_gate * self.tanh(self.W_soc(embs_soc))
        return out

class MLP_0(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_0, self).__init__()
       
        self.linear = nn.Linear(input_dim, output_dim).to('cuda:0')
        
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, Zi, Hi):
        
        concatenated = torch.cat((Zi, Hi), dim=1)
        weight_output = F.softmax(self.leaky_relu(self.linear(concatenated)), dim=1)

        return weight_output
