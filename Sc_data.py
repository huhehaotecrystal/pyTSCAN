from MST import MST

import os
import os.path
import time
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

class Sc_data:
    def __init__(self, expr, pData = None, fData = None):
        self.raw_expr = expr
        self.expr = expr
        if pData is None:
            self.pData = pd.DataFrame({"cell_name" : list(expr.columns)}, index = list(expr.columns))
        else:
            self.pData = pData
        if fData is None:
            self.fData = pd.DataFrame({"gene_name" : list(expr.index)}, index = list(expr.index))
        else:
            self.fData = fData
        self.norm_expr = None
        self.super_expr = None
        self.pca_expr = None
        self.ev = None
        self.evr = None
        self.pc_pres = None
        self.gmm_bics = []
        self.gmm_centroid = []
        self.centroid_order = None
        self.mst_pairs, self.mst_edges = [], []
        self.vijs, self.vij_norms = [], []
        self.__update_nums()
               
    def __update_nums(self):
        self.gene_num = self.expr.shape[0]
        self.cell_num = self.expr.shape[1]
        
    def filter_data(self, min_cells = 2, min_genes = 10):
        """筛选表达矩阵"""
        gene_sum = self.raw_expr.sum(axis = 1)
        cell_sum = self.raw_expr.sum(axis = 0)
        self.expr = self.raw_expr[gene_sum >= min_cells]
        self.fData = self.fData[gene_sum >= min_cells]
        self.expr = self.expr.T[cell_sum >= min_genes].T
        self.pData = self.pData[cell_sum >= min_genes]
        gene_num, cell_num = self.gene_num, self.cell_num
        self.__update_nums()
        gene_num_filtered, cell_num_filtered = self.gene_num, self.cell_num
        print("Filtering done!(%d/%d) cells filtered, (%d/%d) genes filtered." % 
              (cell_num - cell_num_filtered, cell_num, gene_num - gene_num_filtered, gene_num))
    
    def normlize(self, scale_factor = 10000):
        """Seurat标准化，每个细胞除以其总表达量，再乘上scale_factor，再将表达矩阵加1，取自然对数"""
        expr_sum = self.expr.sum(axis = 0)
        self.norm_expr = self.expr / expr_sum * scale_factor
        self.norm_expr = np.log(self.norm_expr + 1)
        
    def preprocessing(self):
        """基因层次聚类"""
        ac = AgglomerativeClustering(n_clusters = self.norm_expr.shape[0] // 20, affinity = "euclidean", linkage = "complete")
        labels = ac.fit_predict(self.norm_expr)
        self.fData["super_gene_cluster"] = labels
        self.super_expr = []
        for cl in np.unique(labels):
            data_part = self.norm_expr[labels == cl]
            self.super_expr.append(data_part.mean(axis = 0))
        self.super_expr = pd.DataFrame(np.array(self.super_expr), columns = self.norm_expr.columns, index = list(np.unique(labels)))
    
    def pca(self, n_components = 20):
        pca = PCA(n_components = n_components)
        self.pca_expr = pca.fit_transform(self.super_expr.T.values)
        self.ev, self.evr = pca.explained_variance_, pca.explained_variance_ratio_
        self.pc_pres = np.diff(self.ev).argmin() + 1
        
    def pcelbow_plot(self):
        plt.scatter([i+1 for i in range(len(self.ev))], self.ev)
        plt.ylabel("explained variance")
        plt.xlabel("PCs")
        
    def gmm(self):
        pca_expr = self.pca_expr[:, :self.pc_pres]
        for n_components in range(2, 30):
            gmm = GaussianMixture(n_components = n_components)
            gmm.fit(pca_expr)
            self.gmm_bics.append(gmm.bic(pca_expr))
        n_components = self.gmm_bics.index(min(self.gmm_bics)) + 1
        gmm = GaussianMixture(n_components = n_components)
        gmm.fit(pca_expr)
        cluster = gmm.predict(pca_expr)
        cluster = cluster + 1
        self.pData["gmm_cluster"] = cluster
    
    def calc_centroid(self):
        pca_expr = self.pca_expr[:, :self.pc_pres]
        for tp in np.unique(self.pData["gmm_cluster"]):
            data_part = pca_expr[self.pData["gmm_cluster"] == tp, :]
            self.gmm_centroid.append(data_part.mean(axis = 0))
        self.gmm_centroid = np.array(self.gmm_centroid)
        
    def __minimumSpanningTree(self):
        """最小生成树Kruskal算法"""
        # 计算簇中心之间的欧氏距离
        pairs, edges = [], []
        point_num = self.gmm_centroid.shape[0]
        for i in range(point_num - 1):
            point_i = self.gmm_centroid[i, :]
            for j in range(i+1, point_num):
                point_j = self.gmm_centroid[j, :]
                pairs.append((i, j))
                edges.append(np.linalg.norm(point_i - point_j))

        # 计算最小生成树
        tree = [[i] for i in range(point_num)]
        while len(tree) > 1:
            min_edge_index = edges.index(min(edges))
            point_i, point_j = pairs[min_edge_index]
            point_pos = self.__judge_point(tree, point_i, point_j)
            if point_pos:  # 两点不在同一个子树上
                i_loc, j_loc = point_pos[0], point_pos[1]
                tree[i_loc].extend(tree.pop(j_loc))
                self.mst_pairs.append(pairs[min_edge_index])
                self.mst_edges.append(edges[min_edge_index])
                pairs.pop(min_edge_index)
                edges.pop(min_edge_index) 
            else:
                pairs.pop(min_edge_index)
                edges.pop(min_edge_index) 
        self.mst_pairs = [(i+1, j+1) for i, j in self.mst_pairs]
        #print("Total distance: %s" % round(sum(self.mst_edges), 3))

    def __judge_point(self, tree, point_i, point_j):
        for count, sub_tree in enumerate(tree):
            if point_i in sub_tree:
                i_loc = count
            if point_j in sub_tree:
                j_loc = count
        if i_loc == j_loc:
            return None
        else:
            return (i_loc, j_loc)
        
    def mst(self):
        cluster_cell_nums = list(self.pData["gmm_cluster"].value_counts().sort_index())
        self.__minimumSpanningTree()
        self.centroid_order = MST(self.mst_pairs).getLongestPath(cluster_cell_nums)
        
    def project_cells(self):
        """
        细胞分为三类：
        1. 位于头尾两群的，直接向连接头尾的两边投射
        2. 位于中间群的，计算细胞与左右相邻两群中心的距离，离哪个近就投射到哪边
        3. 位于不在主路径上的群的，计算细胞与所有其他主路径簇中心的距离，归入最近的主路径簇，再按照第一或第二种细胞处理
        对于细胞排序而言，主要通过“排序簇”、“边”、“值”三列进行排序
        对于在主路径的簇中的细胞，排序簇等于聚类簇；而对于不在主路径簇中的细胞，排序簇等于离其最近的主路径簇
        """
        # calculate edges and norms of edges
        for cent in range(len(self.centroid_order) - 1):
            vij = self.gmm_centroid[self.centroid_order[cent+1] - 1, :] - self.gmm_centroid[self.centroid_order[cent] - 1, :]
            self.vijs.append(vij)
        self.vij_norms = [np.linalg.norm(i) for i in self.vijs]
        
        # project_cells
        cell_order_clusters, cell_order_values, cell_order_edges = [], [], []
        cell_order_types = []  # 记录每个细胞的类型
        pca_expr = self.pca_expr[:, :self.pc_pres]
        for cell_idx in range(self.pData.shape[0]):
            if not self.pData["gmm_cluster"][cell_idx] in self.centroid_order:  # 第三种细胞
                cell_order_types.append("III")
                clustering_cluster = self.pData["gmm_cluster"][cell_idx]
                # 计算细胞与所有主路径簇中心的距离
                distances = [np.linalg.norm(pca_expr[cell_idx, :] - self.gmm_centroid[i-1, :]) for i in self.centroid_order]
                min_index = distances.index(min(distances))
                cell_cluster = self.centroid_order[min_index]        
            else:
                cell_cluster = self.pData["gmm_cluster"][cell_idx]

            if cell_cluster in [self.centroid_order[0], self.centroid_order[-1]]:  # 第一种细胞
                cell_order_types.append("I")
                if cell_cluster == self.centroid_order[0]:  # 细胞位于头簇
                    cell_order_values.append(self.vijs[0].dot(pca_expr[cell_idx, :]) / self.vij_norms[0])
                    cell_order_edges.append(0)
                else:  # 细胞位于尾簇
                    cell_order_values.append(self.vijs[-1].dot(pca_expr[cell_idx, :]) / self.vij_norms[-1])
                    cell_order_edges.append(len(self.vijs) - 1)

            else:  # 第二种细胞
                cell_order_types.append("II")
                path_index = self.centroid_order.index(cell_cluster)
                left_cluster, right_cluster = self.centroid_order[path_index - 1], self.centroid_order[path_index + 1]
                left_distance = np.linalg.norm(pca_expr[cell_idx, :] - self.gmm_centroid[left_cluster - 1, :])
                right_distance = np.linalg.norm(pca_expr[cell_idx, :] - self.gmm_centroid[right_cluster - 1, :])
                if left_distance <= right_distance:  # 离左簇比较近
                    cell_order_values.append(self.vijs[path_index - 1].dot(pca_expr[cell_idx, :]) / self.vij_norms[path_index - 1])
                    cell_order_edges.append(path_index - 1)
                else:  # 离右簇比较近
                    cell_order_values.append(self.vijs[path_index].dot(pca_expr[cell_idx, :]) / self.vij_norms[path_index])
                    cell_order_edges.append(path_index)

            cell_order_clusters.append(cell_cluster)
        self.pData["cell_order_clusters"] = cell_order_clusters
        self.pData["cell_order_values"] = cell_order_values
        self.pData["cell_order_edges"] = cell_order_edges
        self.pData["cell_order_types"] = cell_order_types
        
        # cell ordering
        self.pData["cluster_index"] = self.pData["cell_order_clusters"].apply(lambda x : self.centroid_order.index(x))
        new_pdata = self.pData.sort_values(by = ["cluster_index", "cell_order_edges", "cell_order_values"])
        cell_order_dict = {}
        for i, cell in enumerate(list(new_pdata.index)):
            cell_order_dict[cell] = i + 1
        self.pData["cell_order"] = [cell_order_dict[cell] for cell in list(self.pData.index)]