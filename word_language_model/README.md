# Project 3 assignment
## Training RNN on Windows
At first I directly train the RNN model on my own computer. It told me that my cuda version is too old (cuda 8.0) so I trained it using CPU, which is extremelly slow.

<p align="left">
  <img src="Project 3/picture/epoch1_win.PNG" width=500>
</p>

<p align="left">
  <img src="Project 3/picture/epoch6_win.PNG" width=500>
</p>

For each epoch, it took me more than 2000s. If I want to train 40 epochs, it will take nearly 24h.

In the training process we can see that training loss decreased quickly, validation loss decreased slowly, but they both decreased, demonstrating that the model is training normally. The final test loss is 4.95, which is smalled than traing loss and validation loss. The generated results can be seen [here](Project3/generated_win.txt).

## Training RNN on SCC
I used ssh MobaXterm connecting to SCC.
```
module load python3
module load cuda
module load pytorch
```
I think one of the advantages of using SCC is that you don't need to download various softwares to configure your environment. You just simply load required modules then you can run your program.
```
qrsh -l gpus=1 -l gpu_type=P100
```
Then I applied for a P100 gpu for my training process, which could hold for 12 hours.

You can also submit your program to batches and get results back.
```
qsub -l gpus=1 -l gpu_type=P100 <your command>
```
### Train with 6 epochs
The training speed is totally different from that on windows.

<p align="left">
  <img src="Project 3/picture/epoch1_scc.PNG" width=500>
</p>

<p align="left">
  <img src="Project 3/picture/epoch6_scc.PNG" width=500>
</p>

You can find that each epoch only need averagely 45s, which is nearly 50× speechup. From this I understand the big differences in computing capability between GPU and CPU.

Then I generated [results](Project3/generated_scc_6epochs.txt). 

<p align="left">
  <img src="Project 3/picture/generated_scc.PNG" width=200>
</p>


### Train with 40 epochs
I also trained a slower but better model with 0.5 dropout and 40 epochs which has word embedding size 650 rather than 200 and 650 hidden units per layer instead of 200. The generated results can be found [here](Project3/generated_scc_40epochs.txt).

<p align="left">
  <img src="Project 3/picture/epoch1_40_scc.PNG" width=500>
</p>

<p align="left">
  <img src="Project 3/picture/epoch40_scc.PNG" width=500>
</p>

In the training I find that the learning rate decreased following below rule.

|epochs|1-10|11-18|19-23|24-28|29-37|38-40|
|------|----|-----|-----|-----|-----|-----|
|lr|20|5|1.25|0.31|0.08|0.02|

The learning rate decreased by 4× each time, which means that as the training goes deeper, the learning rate need to slow down to avoid gradient decreasing too fast.

Another thing that we can see is that training loss, validation loss and testing loss are all smaller than results in epochs 6 training, which means our model is better than last one.

[Here](https://www.dropbox.com/s/dzygme0bwcn9ykq/model_scc_40epochs.pt?dl=0) is the pretrained model that I trained.

 
# Word-level language modeling RNN

This example trains a multi-layer RNN (Elman, GRU, or LSTM) on a language modeling task.
By default, the training script uses the Wikitext-2 dataset, provided.
The trained model can then be used by the generate script to generate new text.

```bash 
python main.py --cuda --epochs 6           # Train a LSTM on Wikitext-2 with CUDA
python main.py --cuda --epochs 6 --tied    # Train a tied LSTM on Wikitext-2 with CUDA
python main.py --cuda --epochs 6 --model Transformer --lr 5   
                                           # Train a Transformer model on Wikitext-2 with CUDA
python main.py --cuda --tied               # Train a tied LSTM on Wikitext-2 with CUDA for 40 epochs
python generate.py                         # Generate samples from the trained LSTM model.
python generate.py --cuda --model Transformer
                                           # Generate samples from the trained Transformer model.
```

The model uses the `nn.RNN` module (and its sister modules `nn.GRU` and `nn.LSTM`)
which will automatically use the cuDNN backend if run on CUDA with cuDNN installed.

During training, if a keyboard interrupt (Ctrl-C) is received,
training is stopped and the current model is evaluated against the test dataset.

The `main.py` script accepts the following arguments:

```bash
optional arguments:
  -h, --help            show this help message and exit
  --data DATA           location of the data corpus
  --model MODEL         type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU,
                        Transformer)
  --emsize EMSIZE       size of word embeddings
  --nhid NHID           number of hidden units per layer
  --nlayers NLAYERS     number of layers
  --lr LR               initial learning rate
  --clip CLIP           gradient clipping
  --epochs EPOCHS       upper epoch limit
  --batch_size N        batch size
  --bptt BPTT           sequence length
  --dropout DROPOUT     dropout applied to layers (0 = no dropout)
  --tied                tie the word embedding and softmax weights
  --seed SEED           random seed
  --cuda                use CUDA
  --log-interval N      report interval
  --save SAVE           path to save the final model
  --onnx-export ONNX_EXPORT
                        path to export the final model in onnx format
  --nhead NHEAD         the number of heads in the encoder/decoder of the
                        transformer model
```

With these arguments, a variety of models can be tested.
As an example, the following arguments produce slower but better models:

```bash
python main.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 40           
python main.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied    
python main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40        
python main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40 --tied 
```
