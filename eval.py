### 模型的测试
from model import Model
from evaluator import Evaluator
import torch
import warnings

def _eval(path_to_checkpoint_file, path_to_eval_lmdb_dir, device):
    model = Model()
    model.restore(path_to_checkpoint_file)
    model.to(device)
    accuracy = Evaluator(path_to_eval_lmdb_dir).evaluate(model,device)
    print('Evaluate %s on %s, accuracy = %f' % (path_to_checkpoint_file, path_to_eval_lmdb_dir, accuracy))


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path_to_test_lmdb_dir = r'./data/test.lmdb'
    path_to_checkpoint_file = r'logs/model-66000.pth'
    print('Start evaluating')
    _eval(path_to_checkpoint_file, path_to_test_lmdb_dir,device)
    print('Done')
