<p>
  <img src="figs/Framework_Congrat.jpg" width="1000">
  <br />
</p>

<hr>

<h1> Contrastive Multi-Knowledge Graph Learning for Fake News Detection </h1>

Open-sourced implementation for Contrastive Multi-Knowledge Graph Learning for Fake News Detection - Congrat.

<h2> Heterogeneous graph construction </h2>

Social media platforms categorize the vast volume of published news articles based on their respective topics for efficient management. Hence, topics play a pivotal role as one of the primary properties defining real-world news. Furthermore, it is widely recognized that news articles typically encompass various entities that incorporate compelling and trending information to captivate the readers’ interests. Thus, entities within news articles represent another essential property that contributes to the definition and understanding of news. Since topics and entities play significant roles in expressing the content of news articles, these two properties also offer valuable insights for fake news detection.
In our work, we build a Heterogeneous (HG) with three types of nodes (i.e., news, entities, and topics), where V = {n, t, e} and R = {x_n,t, x_n,e, x_e,e}. n, t, and e denote news nodes, topic nodes, and entity nodes in the HG respectively. xn,t, xn,e and xe,e represent news-topic edges, news-entity edges, and entity-entity edges, detailed in Fig. 1. We propose a series of steps designed to effectively construct the heterogeneous graph including node-level extraction, multi-edge connection, and feature initialization.


<h2> Python Dependencies </h2>

Our proposed Congrat framework is implemented in Python 3.7 and major libraries include: 

* [Pytorch](https://pytorch.org/) = 1.11.0+cu102
* [PyG] (https://pytorch-geometric.readthedocs.io/en/latest/) torch-geometric=2.1.0

More dependencies are provided in requirements.txt.

<h2> To Run </h2>

`python src/main.py`

<h2> Experimental Results </h2>

The experimental results will be open when the article is accepted.
