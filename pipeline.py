from alpha_net import ChessNet, train
from ChessEnv import envModel
from MCTS import MCTSAlphaZero
import os
import pickle
import numpy as np
import torch
import torch.multiprocessing as mp
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__=="__main__":
    for iteration in range(10):
        # Runs MCTS
        net_to_play="current_net_trained8_iter1.pth.tar"
        mp.set_start_method("spawn",force=True)
        net = ChessNet().to(device)
        net.share_memory()
        net.eval()
        print("Run MCTS")
        current_net_filename = os.path.join("./model_data/", net_to_play)
        checkpoint = torch.load(current_net_filename)
        net.load_state_dict(checkpoint['state_dict'])
        MCTS_train = MCTSAlphaZero(envModel, 2, net)
        processes1 = []
        for i in range(6):
            p1 = mp.Process(target=MCTS_train.MCTS_self_play,args=(1, i))
            p1.start()
            processes1.append(p1)
        for p1 in processes1:
            p1.join()
            
        # Runs Net training
        net_to_train="current_net_trained8_iter1.pth.tar"; save_as="current_net_trained8_iter1.pth.tar"
        # gather data
        data_path = "./datasets/iter500/"
        datasets = []
        for idx,file in enumerate(os.listdir(data_path)):
            filename = os.path.join(data_path,file)
            with open(filename, 'rb') as fo:
                datasets.extend(pickle.load(fo, encoding='bytes'))

        data_path = "./datasets/iter300/"
        for idx,file in enumerate(os.listdir(data_path)):
            filename = os.path.join(data_path,file)
            with open(filename, 'rb') as fo:
                datasets.extend(pickle.load(fo, encoding='bytes'))

        data_path = "./datasets/iter1/"
        for idx,file in enumerate(os.listdir(data_path)):
            filename = os.path.join(data_path,file)
            with open(filename, 'rb') as fo:
                datasets.extend(pickle.load(fo, encoding='bytes'))
        
        
        mp.set_start_method("spawn",force=True)
        net = ChessNet().to(device)
        net.share_memory()
        net.train()
        print("Train chess net")
        current_net_filename = os.path.join("./model_data/", net_to_train)
        checkpoint = torch.load(current_net_filename)
        net.load_state_dict(checkpoint['state_dict'])
        net = ChessNet()

        processes2 = []
        for i in range(6):
            p2 = mp.Process(target=train,args=(net, datasets, 100, i))
            p2.start()
            processes2.append(p2)
            
        for p2 in processes2:
            p2.join()
        # save results
        torch.save({'state_dict': net.state_dict()}, os.path.join("./model_data/", save_as))