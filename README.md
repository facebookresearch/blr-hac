
# Code for the associated paper: Bootstrapping Linear Models for Fast Online Adaptation in Human-Agent Collaboration

[Benjamin A. Newman](https://newmanben.com/), [Chris Paxton](https://cpaxton.github.io/about/), [Kris Kitani](https://kriskitani.github.io/), [Henny Admoni](https://hennyadmoni.com/)

[[`Paper`](https://arxiv.org/abs/2404.10733)]

## Requirements  
BLR-HAC is tested on   
* Ubuntu 22  
  
## Installing BLR-HAC
`conda create -n blrhac python=3.8 mpi4py=3.1`  
`pip install -r requirements.txt`  
install pytorch: https://pytorch.org/get-started/locally/  

## Running BLR-HAC  
**Create a dataset**  
`python sample_objectives.py`  
`python sample_sequences.py`  

**Train a model**  
`python trainers.py`  

**Evaluate the model on zero-shot:**  
`python evaluators.py eval_fn=eval_prefs`  

**Evaluate the model on stationary adaptation:**  
For linear and mlp:  
`python evaluators.py eval_fn=eval_online`  

For transformer baseline:  
`python evaluators.py eval_fn=eval_online_sgd`  

**Evaluate the model on nonstationary adaptation:**  
For linear and mlp:  
`python evaluators.py eval_fn=eval_online_switching`  

For transformer baseline:  
`python evaluators.py eval_fn=eval_switching_sgd`  

## Join the BLR-HAC community  
* Website: https://sites.google.com/view/blr-hac  

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.  

## License  
The majority of BLR-HAC is licensed under CC-BY-NC 4.0, as found in the LICENSE file, however portions of the project are available under separate license terms:  

Early Stopping for Pytorch is licensed under the MIT license; Decision Transformer is licensed under the MIT license; Decision Transformer Trajectory Model is licensed under an Apache 2.0 license; and Huggingface Transformer is licensed under an Apache 2.0 license.  

## Citing BLR-HAC  
If you use BLR-HAC in your research please use the following BibTeX entry:  

```
@inproceedings{newman24bootstrapping,
    author = {Newman, Benjamin A. and Paxton, Chris and Kitani, Kris and Admoni, Henny},
    title = {Bootstrapping Linear Models for Fast Online Adaptation in Human-Agent Collaboration},
    year = {2024},
    isbn = {9798400704864},
    publisher = {International Foundation for Autonomous Agents and Multiagent Systems},
    address = {Richland, SC},
    booktitle = {Proceedings of the 23rd International Conference on Autonomous Agents and Multiagent Systems},
    pages = {1463–1472},
    numpages = {10},
    keywords = {assistive robotics, collaborative assistance, human-robot interaction, online assistance},
    location = {, Auckland, New Zealand, },
    series = {AAMAS '24}
}
```
