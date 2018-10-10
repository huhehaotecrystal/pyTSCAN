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

class MST:
    def __init__(self, pairs):
        self.g = {}
        self.nodes = []
        self.node_colors = {}
        self.centroid_order = []  # cluster排序
        self.sub_order = None  # 如果最优路径多于1条，则旁路放置于此
        for p in pairs:
            if not p[0] in self.g:
                self.g[p[0]] = []
                self.nodes.append(p[0])
                self.node_colors[p[0]] = "white"
            if not p[1] in self.g:
                self.g[p[1]] = []
                self.nodes.append(p[1])
                self.node_colors[p[1]] = "white"
            self.g[p[0]].append(p[1])
            self.g[p[1]].append(p[0])
       
    def __longestPath(self, cluster_cell_nums):        
        head = []  # 最长路径一定从叶节点开始
        for node in self.nodes:
            if len(self.g[node]) == 1:
                head.append(node)
        for start in head:
            self.__findAllPaths(start)
            self.__reset_color()
        
        # 路径去重
        new_paths = []
        for path in self.centroid_order:
            if not path in new_paths and not path[::-1] in new_paths:
                new_paths.append(path)
        self.centroid_order = new_paths
        
        # 如果有多于1条的最长路径，选择细胞最多的路径
        if len(self.centroid_order) > 1:
            cell_num_in_paths = []
            for path in self.centroid_order:
                cell_num_in_paths.append(sum([cluster_cell_nums[i-1] for i in path]))
            longest_index = cell_num_in_paths.index(max(cell_num_in_paths))
            self.sub_order = self.centroid_order
            self.centroid_order = [self.sub_order.pop(longest_index)]
        
        # 设定选取小号作为排序起始
        if self.centroid_order[0][0] > self.centroid_order[0][-1]:
            self.centroid_order = [self.centroid_order[0][::-1]]
    
    def getLongestPath(self, cluster_cell_nums):
        self.__longestPath(cluster_cell_nums)
        return self.centroid_order[0]
    
    def __get_color(self, node):
        return self.node_colors[node]
    
    def __set_color(self, node, color):
        self.node_colors[node] = color
        
    def __reset_color(self):
        for node in self.nodes:
            self.node_colors[node] = "white"
            
    def __judge_neighbor_color(self, node):
        neighbors = self.g[node]
        for nei in neighbors:
            if self.__get_color(nei) != "grey":
                return False
        return True
     
    def __findAllPaths(self, start):
        """在最小生成树中遍历出某节点起始的所有路径"""
        res, stack = [], [start]        
        while stack:
            cur_node = stack.pop()
            if self.__get_color(cur_node) == "white":
                self.__set_color(cur_node, "grey")
                res.append(cur_node)
            flag = 0
            for node in self.g[cur_node]:
                if self.__get_color(node) == "white":
                    stack.append(node)
                    flag = 1
            if flag == 0:  # 没有压栈则回溯
                self.centroid_order.append(res)
                tmp = res[:]
                for i in res[::-1]:
                    if self.__judge_neighbor_color(i):
                        tmp.pop()
                    else:
                        break
                res = tmp[:]
