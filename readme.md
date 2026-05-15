# HG-LLM: Dual-granularity cross-modal fusion of hypergraph modeling and LLM explanations for vulnerability detection

## Abstract
With the increasing complexity of software systems, automated vulnerability detection has become a critical task for ensuring software security. Existing graph neural network (GNN)-based approaches can capture structural dependencies in programs, yet their reliance on conventional graph representations limits them to pairwise relations, making it difficult to model higher-order interactions among multiple program elements, such as function calls and loop bodies. Moreover, their capability to capture implicit program semantics remains limited. In contrast, large language models (LLMs) exhibit strong semantic understanding of source code, but lack explicit awareness of program structures, often resulting in unstable predictions when directly applied to vulnerability detection. To address these limitations, we propose HG-LLM, a novel cross-modal vulnerability detection model that integrates hypergraph structural modeling with LLM-enhanced semantic representations. Specifically, we first construct code hypergraphs to explicitly model higher-order structural dependencies among multiple program elements, thereby improving the representation of complex program behaviors. We then employ an LLM to generate functional explanations of source code and encode them into continuous semantic representations to enhance the detection of semantically related vulnerabilities. Finally, we design a dual-granularity cross-modal fusion mechanism that performs deep interaction between structural and semantic representations at both node and graph levels, enabling effective collaboration between hypergraph structural modeling and LLM semantic understanding. Experiments on three public datasets, FFmpeg+Qemu, Reveal, and Big-Vul, demonstrate that HG-LLM consistently outperforms state-of-the-art baselines in terms of accuracy and F1-score. Ablation studies and visualization analyses further verify the complementary benefits of higher-order structural representation and deep semantic understanding.

## Dataset
Experiments are conducted on three widely used public vulnerability detection datasets: FFmpeg+Qemu, Reveal, and Big-Vul. 

1. Fan et al. [1]: https://drive.google.com/file/d/1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X/view?usp=sharing
2. Reveal [2]: https://drive.google.com/drive/folders/1KuIYgFcvWUXheDhT--cBALsfy1I4utOyF
3. FFMPeg+Qemu [3]: https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF

## Requirement 
Our code is based on Python3 (>= 3.7). There are a few dependencies to run the code. The major libraries are listed as follows:
- torch == 1.9.1+cu111
- torchaudio == 0.9.1
- torchvision == 0.10.1+cu111
- dgl-cuda11.1 == 0.9.1
- numpy == 1.21.6
- scikit-learn == 1.0.2
- transformers == 4.25.1
- pandas == 1.3.5
- graphviz == 2.50.0
- gensim == 4.2.0
- tqdm == 4.64.1

Default settings in HG_LLM:
>batch_size = 64
>lr = 5e-5
>weight_decay=1e-4
>betas=(0.9, 0.98)
>epoch = 100
>patience = 30


## Preprocessing 
For different datasets,  we need run the data processing code, see the `data_processing` folder for details. The detailed steps are as follows：
1. data_preprocess.py: Loads the dataset and generates '.c' files
2. process_joern.py: Batch-processes calls to joern to generate CPG
3. train_word2vec.py: Pre-trains word vectors
4. comment.py: Call LLM to generate semantic information
5. cpg_gs.py: Performs graph simplification and generates hypergraphs

Modify the directory in the relevant configuration, then run `main.py`

## References
[1] Y. Li, S. Wang, and T. N. Nguyen, “Vulnerability detection with fine-grained interpretations,” in ESEC/FSE ’21: 29th ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering, Athens, Greece, August 23-28, 2021. 
[2] S. Chakraborty, R. Krishna, Y. Ding, and B. Ray, “Deep learning based vulnerability detection: Are we there yet?” CoRR, vol. abs/2009.07235,2020.
[3] Y. Zhou, S. Liu, J. K. Siow, X. Du, and Y. Liu, “Devign: Effective vulner-ability identification by learning comprehensive program semantics via graph neural networks,” in Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems. 2019, NeurIPS 2019, 2019, pp. 10 197–10 207.