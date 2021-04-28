import argparse

import torch
from torch.utils import data as torch_data
import torch.optim as optim
from torch.utils import tensorboard

from model.net import Net
import utils
from model.data_loader import data_loader
from build_vocab import create_corpus

parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--dataset_path', default='./data',
                    help="Path to dataset.")
parser.add_argument('--checkpoint_path', default='./experiments/checkpoint',
                    help="Path to model checkpoint (by default train from scratch).")
parser.add_argument('--tensorboard_log_dir', default='./experiments/log',
                    help="Path for tensorboard log directory.")

if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    
    params = utils.Params(json_path)
    params.device = torch.cuda.is_available()
    
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # dataset
    text, word2idx, idx2word, word_freqs, word_counts = create_corpus('./questions.txt')
    train_set = data_loader.DataLoader(text, word2idx, idx2word, word_freqs, word_counts)
    dataloader = torch_data.DataLoader(train_set, batch_size=params.batch_size, shuffle=True)
    print("corpus: {}".format(len(idx2word)))
    del text
    
    # model 
    model = net.Net(vocab_size=len(idx2word), emb_size=params.embedding_dim).cuda() if params.cuda else net.Net(params)
    optimizer = optim.SGD(model.parameters(), lr = params.learning_rate)
    model.train()
    model = model.to(params.device)
    summary_writer = tensorboard.SummaryWriter(log_dir=args.tensorboard_log_dir)
    step = 0
    start_epoch_id = 1
    best_score=0
    print(model)

    # training
    for epoch_id in range(start_epoch_id, params.epochs + 1):
        print("Epoch {}/{}".format(epoch_id, params.epochs))
        loss = 0
        with tqdm(total=len(dataloader)) as t:
            for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):
                input_labels = input_labels.long().to(device)
                pos_labels = pos_labels.long().to(device)
                neg_labels = neg_labels.long().to(device)
                optimizer.zero_grad()
                loss = model(input_labels, pos_labels, neg_labels).mean()
                loss.backward()

                summary_writer.add_scalar('Loss/train', loss.data.cpu().numpy(), global_step=step)

                optimizer.step()

            t.set_postfix(loss = loss.mean().data.cpu().item())
            t.update()

        if epoch_id % 50 == 0:          
            utils.save_checkpoint(checkpoint_path, model, optimizer, epoch_id, step, loss)
    
        

    # embedding_weights = model.input_embeddings()
    # torch.save(model.state_dict(), "embedding-{}.th".format(EMBEDDING_SIZE))
