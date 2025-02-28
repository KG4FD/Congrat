<p>
  <img src="figs/Framework_Congrat.jpg" width="1000">
  <br />
</p>

<hr>

<h1> Contrastive Multi-Knowledge Graph Learning for Fake News Detection </h1>

Open-sourced implementation for Contrastive Multi-Knowledge Graph Learning for Fake News Detection - Congrat.

<h2> Heterogeneous graph construction </h2>

Social media platforms categorize the vast number of published news articles based on their respective topics to facilitate efficient management. As a result, topics serve as a fundamental characteristic that defines real-world news. Additionally, news articles often incorporate various entities that reflect trending and compelling information, capturing readers' attention. These entities play a crucial role in shaping the content and context of news, making them another essential property for understanding news articles. Given the significance of both topics and entities in representing news content, these properties also provide valuable insights for fake news detection.

In this work, we construct a Heterogeneous Graph (HG) that consists of three types of nodes: news, entities, and topics. Formally, we define the node set as 𝑉={𝑛, 𝑡, 𝑒} and the relationship set as 𝑅= {𝑥_𝑛,𝑡, 𝑥_𝑛,𝑒, 𝑥_𝑒,𝑒}, where 𝑛, 𝑡, and 𝑒 represent news nodes, topic nodes, and entity nodes, respectively. The edges 𝑥_𝑛,𝑡, 𝑥_𝑛,𝑒, and 𝑥_𝑒,𝑒 correspond to news-topic connections, news-entity connections, and entity-entity connections, as illustrated in Fig. 1. To effectively construct this heterogeneous graph, we propose a structured process consisting of node-level extraction, multi-edge connection, and feature initialization, ensuring a well-defined representation for downstream fake news detection tasks.

1. Node-Level Extraction
In our constructed Heterogeneous Graph (HG), each news article is represented as a news node with assigned labels. To generate entity nodes, we employ TAGME 2, an entity linking tool, to extract various entities present within the news content. For topic nodes, we utilize the Latent Dirichlet Allocation (LDA) model to identify the underlying topics of each news article. Specifically, we set a threshold of 𝛼=50 to define the total number of topics in the LDA model and assign each news article its top 2 most relevant topics.

2. Multi-Edge Connection
After constructing the nodes in the Heterogeneous Graph (HG), the next step is to establish connections between them. We define three types of edges in the HG:

News-Topic Edges (𝑥_𝑛,𝑡): A news-topic edge is formed when a news article is associated with a specific topic. Each news node is connected to K topic nodes.

News-Entity Edges (𝑥_𝑛,𝑒): A news-entity edge is created when a news article contains a particular entity. Each news node can be linked to multiple entity nodes.

Entity-Entity Edges (𝑥_𝑒,𝑒): An entity-entity edge is established when the cosine similarity score between two entities exceeds a predefined threshold of 𝛽=0.5.

These connections ensure a structured and meaningful representation of relationships within the HG, facilitating effective fake news detection.

3. Feature Initialization
We apply various natural language processing techniques, including Doc2vec, term frequency-inverse document frequency (TF-IDF), and one-hot encoding, to initialize the feature vectors for the nodes in the Heterogeneous Graph (HG). For news nodes, both TF-IDF and Doc2vec are utilized to encode the content of each news article. For entity nodes and topic nodes, TF-IDF and one-hot encoding are used to initialize their respective feature vectors. These methods effectively capture the rich and diverse features inherent in the three types of HG nodes, providing a robust foundation for further processing and analysis.

Finally, we apply the same processing to all four datasets, constructing four distinct Heterogeneous Graphs (HGs) based on the news articles from each dataset, as illustrated in the figure below.

<p>
  <img src="figs/Framework_Congrat.jpg" width="1000">
  <br />
</p>


<h2> Python Dependencies </h2>

Our proposed Congrat framework is implemented in Python 3.7 and major libraries include: 

* [Pytorch](https://pytorch.org/) = 1.11.0+cu102
* [PyG] (https://pytorch-geometric.readthedocs.io/en/latest/) torch-geometric=2.1.0

More dependencies are provided in requirements.txt.

<h2> To Run </h2>

`python src/main.py`

<h2> Experimental Results </h2>

The experimental results will be open when the article is accepted.
