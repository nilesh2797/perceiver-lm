import pickle
with open("deepmind_assets/language_perceiver_io_bytes.pickle", "rb") as f:
    params = pickle.loads(f.read())
from perceiver_io.perceiver_im import PerceiverLM
#from data import MNISTDataModule
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


transform_train = transforms.Compose([
transforms.RandomCrop(32, padding=4),
transforms.RandomHorizontalFlip(),
transforms.ToTensor(),
transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
NUM_CLASSES=10
trainset = torchvision.datasets.STL10(root='./data',  split='train',
                                          download=False, transform=transform_train)
testset = torchvision.datasets.STL10(root='./data', split='test',
                                          download=False, transform=transform_train)
trainset, validset = torch.utils.data.random_split(trainset, 
                                                      [int(len(trainset)*0.8),len(trainset)- 
                                                      int(len(trainset)*0.8)])
dims = (3, 32, 32)
batch_size = 8
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size,
                                            shuffle=False,num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False, num_workers=2)
model = PerceiverLM(vocab_size=262, max_seq_len=2048, embedding_dim=768, num_latents=256, latent_dim=1280,qk_out_dim=256, num_self_attn_per_block=26)
model = model.to('cuda')
state_dict = {}
model_enc_base = 'perceiver.encoder.'
params_enc_base = 'perceiver_encoder/~/'

state_dict['token_embedding.weight'] = params['embed']['embeddings']
state_dict['decoder_token_bias'] = params['embedding_decoder']['bias']
state_dict['position_embedding.weight'] = params['trainable_position_encoding']['pos_embs']
state_dict['query_embedding.weight'] = params['basic_decoder/~/trainable_position_encoding']['pos_embs']
state_dict[f'{model_enc_base}latents'] = params[f'{params_enc_base}trainable_position_encoding']['pos_embs']
state_dict['linear0.weight'] = torch.rand(768,1568).to('cuda')
state_dict['linear0.bias'] = torch.rand(768).to('cuda')
state_dict['linear1.weight'] = torch.rand(10,25152).to('cuda')
state_dict['linear1.bias'] = torch.rand(10).to('cuda')

def copy_attention_params(model_base, params_base):
    global state_dict
    state_dict[f'{model_base}attention.q.weight'] = params[f'{params_base}attention/linear']['w'].T
    state_dict[f'{model_base}attention.q.bias'] = params[f'{params_base}attention/linear']['b']
    state_dict[f'{model_base}attention.k.weight'] = params[f'{params_base}attention/linear_1']['w'].T
    state_dict[f'{model_base}attention.k.bias'] = params[f'{params_base}attention/linear_1']['b']
    state_dict[f'{model_base}attention.v.weight'] = params[f'{params_base}attention/linear_2']['w'].T
    state_dict[f'{model_base}attention.v.bias'] = params[f'{params_base}attention/linear_2']['b']
    state_dict[f'{model_base}attention.projection.weight'] = params[f'{params_base}attention/linear_3']['w'].T
    state_dict[f'{model_base}attention.projection.bias'] = params[f'{params_base}attention/linear_3']['b']

    if 'self_attention' in params_base:
        state_dict[f'{model_base}layer_norm.weight'] = params[f'{params_base}layer_norm']['scale']
        state_dict[f'{model_base}layer_norm.bias'] = params[f'{params_base}layer_norm']['offset']
        state_dict[f'{model_base}qkv_layer_norm.weight'] = params[f'{params_base}layer_norm_1']['scale']
        state_dict[f'{model_base}qkv_layer_norm.bias'] = params[f'{params_base}layer_norm_1']['offset']
    else:
        state_dict[f'{model_base}q_layer_norm.weight'] = params[f'{params_base}layer_norm']['scale']
        state_dict[f'{model_base}q_layer_norm.bias'] = params[f'{params_base}layer_norm']['offset']
        state_dict[f'{model_base}kv_layer_norm.weight'] = params[f'{params_base}layer_norm_1']['scale']
        state_dict[f'{model_base}kv_layer_norm.bias'] = params[f'{params_base}layer_norm_1']['offset']
        state_dict[f'{model_base}qkv_layer_norm.weight'] = params[f'{params_base}layer_norm_2']['scale']
        state_dict[f'{model_base}qkv_layer_norm.bias'] = params[f'{params_base}layer_norm_2']['offset']

    state_dict[f'{model_base}mlp.mlp.0.weight'] = params[f'{params_base}mlp/linear']['w'].T
    state_dict[f'{model_base}mlp.mlp.0.bias'] = params[f'{params_base}mlp/linear']['b']
    state_dict[f'{model_base}mlp.mlp.2.weight'] = params[f'{params_base}mlp/linear_1']['w'].T
    state_dict[f'{model_base}mlp.mlp.2.bias'] = params[f'{params_base}mlp/linear_1']['b']

copy_attention_params(f'{model_enc_base}cross_attn.', f'{params_enc_base}cross_attention/')
copy_attention_params(f'perceiver.decoder.cross_attention.', f'basic_decoder/cross_attention/')
for i in range(26):
    copy_attention_params(f'{model_enc_base}self_attention_block.{i}.', f'{params_enc_base}self_attention{"_%d"%i if i else ""}/')
    
state_dict = {k: torch.tensor(v).to('cuda') for k,v in state_dict.items()}
#state_dict = state_dict.to('cuda')
model.load_state_dict(state_dict)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3 , momentum=0.9, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss().to('cuda')
NUM_EPOCHS = 25
device = 'cuda'
def train(epoch):
    #for epoch in range(NUM_EPOCHS):
        model.train()
        print("model is ", model)
        correct_images = 0
        total_images = 0
        training_loss = 0
        LRlistIteration = []
        trainLossIteration = []
        for batch_index, (images, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            _, predicted = outputs.max(1)
            total_images += labels.size(0)
            correct_images += predicted.eq(labels).sum().item()
            print('Epoch: %d, Batch: %d, Loss: %.3f, '
                            'Accuracy: %.3f%% (%d/%d)' % (epoch, batch_index, training_loss/(batch_index+1),
                                                    100.*correct_images/total_images, correct_images, total_images))
            trainLossIteration.append(training_loss/(batch_index+1))
        return training_loss/(batch_index+1), trainLossIteration

def validate(epoch):
    #for epoch in range(NUM_EPOCHS):
        model.eval()
        validation_running_loss = 0.0
        total_images = 0
        correct_images = 0
        valLossIteration = []
        with torch.no_grad():
            for batch_index, (images, labels) in enumerate(validloader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                validation_running_loss += loss.item()
                _, predicted = outputs.max(1)
                total_images += labels.size(0)
                correct_images += predicted.eq(labels).sum().item()
                print('Epoch: %d, Batch: %d, Loss: %.3f, '
                        'Accuracy: %.3f%% (%d/%d)' % (epoch, batch_index, validation_running_loss/(batch_index+1),
                                                100.*correct_images/total_images, correct_images, total_images))
                valLossIteration.append(validation_running_loss/(batch_index+1))
                epoch_loss = validation_running_loss / (batch_index+1)
        return epoch_loss, valLossIteration

def test():
    model.eval()
    #print(torch.tensor(inputs).size())
    test_loss = 0
    total_images = 0
    correct_images = 0
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model.forward(torch.tensor(images.to(device)))
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total_images += labels.size(0)
            correct_images += predicted.eq(labels).sum().item()
            print(batch_index, len(testloader), 'Loss: %.3f | Accuracy: %.3f%% (%d/%d)'
                    % (test_loss/(batch_index+1), 100.*correct_images/total_images, correct_images, total_images))
            test_accuracy = 100.*correct_images/total_images
    print("accuracy of test set is",test_accuracy)
epoch = NUM_EPOCHS
#for epoch in range(NUM_EPOCHS):
   #     train(epoch)
  #      print("validation loss -------------")
 #       validate(epoch)
#print("test loss -------------")
#test()
train_loss, val_loss = [], []
#learning_rate_per_iteration = []
#learning_rate_per_epoch = []
LRlistIteration = []
train_loss_per_iteration = []
val_loss_per_iteration = []
for epoch in range(NUM_EPOCHS):
  training_epoch_loss, trainLossIteration = train(epoch) 
  #In 1 epoch 48% accuracy
  validation_epoch_loss, valLossIteration =validate(epoch)
  train_loss.append(training_epoch_loss)
  val_loss.append(validation_epoch_loss)
  #learning_rate_per_epoch.append(LR)
  train_loss_per_iteration.extend(trainLossIteration)
  val_loss_per_iteration.extend(valLossIteration)
  #learning_rate_per_iteration.extend(LRs)
  #if warm_restarts == False:
   # scheduler.step() # take default MultiStepLR
  #print(f"[INFO]: Current LR [Epoch end]: {scheduler.get_last_lr()}")
  print(f"Training loss: {training_epoch_loss:.3f}")
  print(f"Validation loss: {validation_epoch_loss:.3f}")
  print('-----------------------***---------------------------')
print("test loss ----------")
test()
#if args.local_rank == 0:
 #       _logger.info('Scheduled epochs: {}'.format(num_epochs))
  #  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
   #                                  std=[0.229, 0.224, 0.225])
    #transform_train = transforms.Compose([
     #       transforms.RandomResizedCrop(224),
      #      transforms.RandomHorizontalFlip(),
       #     transforms.ToTensor(),
        #    normalize,
       # ])
    # create the train and eval datasets
    #dataset_total = torchviscon.datasets.Caltech101(root=args.data_dir, download=True, transform=transform_train)
    #train_size = int(0.8 * len(dataset_total))
    #test_size = len(full_dataset) - train_size
    #dataset_train, dataset_eval = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    #dataset_eval = torchvision.datasets.Caltech101(root=args.data_dir, split='test',download=True, transform=transform_train)
    #dataset_train = torchvision.datasets.STL10(root=args.data_dir, split='train', download=True, transform=transform_train)
    #dataset_eval = torchvision.datasets.STL10(root=args.data_dir, split='test', download=True, transform=transform_train)	
    #path_caltech101 = args.data_dir + '101_ObjectCategories/'
    #totalset = torchvision.datasets.ImageFolder(path_caltech101, transform=transform_train)
    #X, y = zip(*totalset)
    #X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2,stratify=y)
    #dataset_train, dataset_eval = list(zip(X_train, y_train)), list(zip(X_val, y_val))
    #dataset_train = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
    #dataset_train = create_dataset(
     #   args.dataset,
      #  root=args.data_dir, split=args.train_split, is_training=True,
       # batch_size=args.batch_size, repeats=args.epoch_repeats)
    #dataset_eval = torchvision.datasets.CIFAR10(root=args.data_dir, train=False,download=True, transform=transform_train)
    #dataset_eval = create_dataset(
     #   args.dataset, root=args.data_dir, split=args.val_split, is_training=False, batch_size=args.batch_size)

    # setup mixup / cutmix
    #collate_fn = None
    #mixup_fn = None
    #mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    #if mixup_active:
     #   mixup_args = dict(
         #   mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
          #  prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
           # label_smoothing=args.smoothing, num_classes=args.num_classes)
        #if args.prefetcher:
         #   assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
          #  collate_fn = FastCollateMixup(**mixup_args)
        #else:
         #   mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    #if num_aug_splits > 1:
     #   dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)
