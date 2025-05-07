from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.loader import DataLoader
from pathlib import Path
import sys
import os
import pandas as pd
import numpy as np
root_dir=str(Path(__file__).resolve().parents[1])
sys.path.append(root_dir)
os.chdir(root_dir)
from synprop.graphdataset_5 import ReactionDataset #or graphdataset 0/2/3/4/5 cho new, graphdataset_1 cho stone age

class data_wrapper_7(object):
    def __init__(self,data_path,graph_path,target,batch_size,num_workers,valid_size,test_size,):
        super(object,self).__init__()
        self.data_path=data_path
        self.graph_path=graph_path
        self.target=target
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.valid_size=valid_size
        self.test_size=test_size
    
    def get_data_loaders(self):
        train_dataset=ReactionDataset(self.data_path,self.graph_path,self.target)
        train_dataset,valid_dataset,test_dataset=self.get_train_val_data_loaders(train_dataset)

        return train_dataset,valid_dataset,test_dataset
    
    def get_train_val_data_loaders(self,train_dataset):

        num_train=len(train_dataset)
        indices=list(range(num_train))
        np.random.shuffle(indices)

        split1=int(np.floor(self.valid_size*num_train))
        split2=int(np.floor(self.test_size*num_train))
        val_idx, test_idx, train_idx = indices[:split1], indices[split1:split1+split2], indices[split1+split2:]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(val_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=train_sampler,
            num_workers=self.num_workers, drop_last=False
        )
        valid_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
            num_workers=self.num_workers, drop_last=False
        )
        test_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=test_sampler,
            num_workers=self.num_workers, drop_last=False
        )

        return train_loader,valid_loader,test_loader

def main():
    from pathlib import Path
    import sys
    import os
    root_dir = str(Path(__file__).resolve().parents[1])
    os.chdir(root_dir)
    sys.path.append(root_dir)
    from synprop.data_wrapper_7 import data_wrapper_7
    data_path='./Data/regression/rad6re/rad6re.csv'
    graph_path='./Data/regression/rad6re/rad6re.pkl.gz'
    target='dh'
    batch_size=2
    num_workers=0
    valid_size=0.1
    test_size=0.1
    data_wrapper=data_wrapper_7(data_path,graph_path,target,batch_size,num_workers,valid_size,test_size)
    train_loader,valid_loader,test_loader=data_wrapper.get_data_loaders()
    print(train_loader)
    print(valid_loader)
    print(test_loader.dataset[0].x.shape)
    print(test_loader.dataset[0].edge_attr.shape)

if __name__=='__main__':
    main()
