import torch
import torch.nn as nn
import datetime
import pytz
import pickle
import time
import subprocess
import fitz
from PIL import Image
import re
import os


class TextGenerator(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers=1, time=True):
        super(TextGenerator, self).__init__()
        self.name = 'GRU'

        if time:
          timezone = pytz.timezone('America/Toronto')
          current_datetime = datetime.datetime.now(timezone)
          date_time_string = current_datetime.strftime("%Y-%m-%d-%H:%M")
          self.name = self.name + '_' + date_time_string

        # identiy matrix for generating one-hot vectors
        self.ident = torch.eye(vocab_size)

        # recurrent neural network
        self.rnn = nn.GRU(vocab_size, hidden_size, n_layers, batch_first=True)

        # a fully-connect layer that outputs a distribution over
        # the next token, given the RNN output
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, inp, hidden=None):
        inp = self.ident[inp]                  # generate one-hot vectors of input
        output, hidden = self.rnn(inp, hidden) # get the next output and hidden state
        output = self.decoder(output)          # predict distribution over next tokens
        return output, hidden
    

def min_max_sequence(model, start_text ,min_len=5,max_len=20, temperature=0.8,must_contain = ""):
    generated_sequence = ""
    inp = [vocab_stoi["<BOS>"]]
    idx,prefix_length,prefix_lst = -1,-1,[]


    if start_text != "":
      idx = 0
      prefix_lst = start_text.split(" ")
      prefix_length = len(prefix_lst)
      # generated_sequence = prefix_lst[idx] + ' '
      # inp = [vocab_stoi[idx]]
      # idx += 1
    if prefix_length > max_len:
      print("Long user input!")
      return
    inp = torch.Tensor(inp).long()
    hidden = None



    for p in range(max_len):
        output, hidden = model(inp.unsqueeze(0), hidden)

        if idx < prefix_length:
          predicted_char = prefix_lst[idx]
          idx += 1
          inp = torch.Tensor([vocab_stoi[predicted_char]]).long()
        elif p == min_len and must_contain != "":
          predicted_char = must_contain
          inp = torch.Tensor([vocab_stoi[predicted_char]]).long()
        else:
          predicted_char = "<pad>"
          while (p < min_len and predicted_char == "<EOS>") or predicted_char in ["<pad>","<BOS>","<unk>"]:
            # Sample from the network as a multinomial distribution
            output_dist = output.data.view(-1).div(temperature).exp()
            top_i = int(torch.multinomial(output_dist, 1)[0])
            # Add predicted character to string and use as next input
            predicted_char = vocab_itos[top_i]

            inp = torch.Tensor([top_i]).long()

        if predicted_char == "<EOS>":

          break

        generated_sequence += predicted_char + ' '

    return generated_sequence


def check_latex_syntax(latex_expression):
  
    curr_dir = os.getcwd()
    temp_dir = os.path.join(curr_dir, 'temp')
    os.chdir(temp_dir)

    temp_tex_file = "temp.tex"
    with open(temp_tex_file,'w') as f:
        f.write(r'''
\documentclass{article}
\begin{document}
$''')
        f.write(latex_expression)
        f.write(r'''$
\end{document}''')
    try:
        subprocess.check_call(['pdflatex', '-interaction=nonstopmode', temp_tex_file])
    except:
        os.chdir(curr_dir)
        return False

    os.chdir(curr_dir)
    return True


def latex2img(latex_expression, file_name):

    curr_dir = os.getcwd()
    temp_dir = os.path.join(curr_dir, 'temp')
    os.chdir(temp_dir)

    temp_tex_file = "temp.tex"
    with open(temp_tex_file, 'w') as f:
        f.write(r'''
\documentclass{article}
\usepackage{amsmath} % Add necessary packages
\begin{document}
$''')
        f.write(latex_expression)
        f.write(r'''$
\end{document}''')
    try:
        subprocess.check_call(['pdflatex', '-interaction=nonstopmode', temp_tex_file])
    except subprocess.CalledProcessError:
        return False
    # Convert generated pdf to png if latex is valid
    try:
      pdf_doc = fitz.open("temp.pdf")  # Open temporary pdf with PyMuPDF
      page = pdf_doc[0]  # Assuming the first page contains the Latex expression
      pix = page.get_pixmap(matrix=fitz.Matrix(5.0, 5.0))  # Render at 5x resolution for better quality
      os.chdir(curr_dir)
      pix.save(file_name)  # Save the rendered image
      pdf_doc.close()
    except Exception as e:
      print(f"Error converting pdf to image: {e}")
      os.chdir(curr_dir)
      return False
    return True

def crop_image(path, i):
    img = Image.open(path + str(i) + '_raw.png')
    left, top, right, bottom = 700, 500, 1500, 800
    cropped_img = img.crop((left, top, right, bottom))
    cropped_img.save(path + str(i) + '.png')

def check_vocab(input_string):

    # Regular expression to match LaTeX commands
    latex_pattern = re.compile(r'\\(?:[^\s\\\{\}]+|\s+|\{[^{}]*\}|\\\n|[{}])')

    # Tokenize the input string in LaTeX format
    input_tokens = latex_pattern.findall(input_string)
    
    # Check if all tokens are in the vocabulary
    for token in input_tokens:
        if token not in vocab_stoi:
            return False
    return True

curr_dir = os.getcwd()
vocab_path = os.path.join(curr_dir, 'src', 'vocab.pkl')

with open(vocab_path, 'rb') as f:
    loaded_vocab = pickle.load(f)
    vocab_stoi = loaded_vocab['stoi']
    vocab_itos = loaded_vocab['itos']
    vocab_size = loaded_vocab['size']

model = TextGenerator(vocab_size, 256)
model_path = os.path.join(curr_dir, 'src', r'GRU_2024-04-04-16_22_bs10_lr0.001_epoch5.zip')
state = torch.load(model_path)
model.load_state_dict(state)


def smart_generator(model, start_text, min_len=5, max_len=20, 
                    temperature=0.8, num_of_img=5, must_contain = ""):
    ret =[]
    diff = max_len - min_len
    len_lwm,len_upm = min_len,min_len
    ranges = diff / num_of_img
    for i in range(num_of_img):
      len_upm += ranges
      found,force = False, False
      start_time,now = time.time(),time.time()
      while not found:
        seq = min_max_sequence(model, start_text, int(len_lwm), int(len_upm), 
                               temperature,must_contain)
        if not force and now - start_time > 1:
          start_text += ' '+ must_contain
          must_contain = ""
          force = True
        if check_latex_syntax(seq):
          found = True
      now = time.time()
      print(now - start_time)
      ret.append(seq)
      len_lwm += ranges
    return ret


def generate(start, min_length=5, max_length=20, required_char=''):

    output = [(None, None), 
              (None, None), 
              (None, None), 
              (None, None), 
              (None, None)]
    
    temps = []
    temps = smart_generator(model, start, min_length, max_length, 0.8, 5, required_char)

    
    for i in range (5):

        latex2img(temps[i], 'img/'+str(i)+'_raw.png')
        crop_image('img/', i)
        output[i] = (temps[i], 'img/'+str(i)+'.png')

    return output