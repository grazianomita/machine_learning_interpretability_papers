# Model-Interpretability
This is a public collection of papers related to machine learning model interpretability.


## Association Rule Mining
Association Rule Mining methods come from the Data Mining community. The so called class association rules are extracted from the data through algorithms like Apriori, FP-growth. A subset of them is selected with simple ranking or selection heuristics, and used to classify new records. Rules are usually expressed in disjunctive normal form (DNF) and are considered to be interpretable by nature. Few and compact rules are preferred.

* [Integrating Classification and Association Rule Mining, KDD 1998](https://www.aaai.org/Papers/KDD/1998/KDD98-012.pdf)
* [Mining the Most Interesting Rules, KDD 1999](https://www.bayardo.org/ps/kdd99.pdf)
* [Growing Decision Trees on Support-Less Association Rules, KDD 2000](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.443.572&rep=rep1&type=pdf)
* [CMAR: Accurate and Efficient Classification Based on Multiple Class-Association Rules, ICDM 2001](http://hanj.cs.illinois.edu/pdf/cmar01.pdf)
* [CPAR: Classification Based on Predictive Association Rules, ICDM 2003](http://hanj.cs.illinois.edu/pdf/sdm03_cpar.pdf)
* [HARMONY: Efficiently Mining the Best Rules for Classification, SIAM 2005](https://pdfs.semanticscholar.org/5ea0/6fb5591b6a32c87a74d09ef0b816805fb8eb.pdf)
* [A new approach to classification based on association rule mining, JDSS 2006](http://www.paper.edu.cn/scholar/showpdf/NUD2EN5INTA0eQxeQh)
* [Discriminative Frequent Pattern Analysis for Effective Classification, ICDE 2007](http://hanj.cs.illinois.edu/pdf/icde07_hcheng.pdf)


## Rule Learning
Here, we report just a subset of the work done on rule-based methods, trying to cover both old and more recent methods.

* [Incremental reduced error pruning, ICML 1994](https://www.researchgate.net/publication/2271856_Incremental_Reduced_Error_Pruning)
* [Fast Effective Rule Induction, ICML 1995](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.107.2612&rep=rep1&type=pdf)
* [A Simple, Fast, and Effective Rule Learner, AAAI 1999](https://pdfs.semanticscholar.org/7237/70d9ac418e923db5e087ae18c04702f5986e.pdf)
* [Disjunctions of Conjunctions, Cognitive Simplicity, and Consideration Sets, Journal of Marketing Research 2010](https://pdfs.semanticscholar.org/bf65/dc3164be43919088695d7f43ee2e51d4b614.pdf)
* [The Set Covering Machine, JMLR 2002](http://www.jmlr.org/papers/volume3/marchand02a/marchand02a.pdf)
* [An Integer Optimization Approach to Associative Classification, NIPS 2012](https://pdfs.semanticscholar.org/ec2c/b3a9abdcad58ea4acb73b1411f7eb98d8472.pdf)
* [Exact Rule Learning via Boolean Compressed Sensing, ICML 2013](http://ssg.mit.edu/~krv/pubs/MalioutovV_icml2013.pdf)
* [Box Drawings for Learning with Imbalanced Data, KDD 2014](https://arxiv.org/abs/1403.3378)
* [Falling Rule Lists, AISTATS 2015](https://arxiv.org/abs/1411.5899)
* [Bayesian Or’s of And’s for Interpretable Classification with Application to Context Aware Recommender Systems, 2015](https://finale.seas.harvard.edu/files/finale/files/techreportboa_wangetal.pdf)
* [nterpretable classifiers using rules and Bayesian analysis: Building a better stroke prediction model, Annals of Applied Statistics 2015](https://arxiv.org/abs/1511.01644)
* [Interpretable Decision Sets: A Joint Framework for Description and Prediction, KDD 2016](https://www-cs-faculty.stanford.edu/people/jure/pubs/interpretable-kdd16.pdf)
* [Interpretable Two-level Boolean Rule Learning for Classification, ICML 2016](https://arxiv.org/abs/1606.05798)
* [A Bayesian Framework for Learning Rule Sets for Interpretable Classification, JMLR 2017](http://jmlr.org/papers/v18/16-003.html)
* [Learning Certifiably Optimal Rule Lists for Categorical Data, JMLR 2018](https://arxiv.org/abs/1704.01701)
* [An Optimization Approach to Learning Falling Rule Lists, AISTATS 2018](http://proceedings.mlr.press/v84/chen18a/chen18a.pdf)
* [Boolean Decision Rules via Column Generation, NIPS 2018](https://arxiv.org/abs/1805.09901)


## Incremental Rule Learning
Incremental methods allow to continuously update the model when new data arrives. For rule-based models (I will also consider decision trees here) updates corresponds to changes in the set of rules (add/remove/update a rule). Research in incremental interpretable methods had a huge impact in 90's, but interest in the field has grown again recently especially in the database community.

* [Incremental Learning from Noisy Data, Machine Learning 1986](https://link.springer.com/content/pdf/10.1007/BF00116895.pdf)
* [Learning in the Presence of Concept Drift and Hidden Contexts, Machine Learning 1996](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.9.9119&rep=rep1&type=pdf)
* [Maintaining the performance of a learned classifier under concept drift, Intelligent Data Analysis 1999](https://www.sciencedirect.com/science/article/pii/S1088467X99000335)
* [Mining High-speed Data Streams, KDD 2000](https://homes.cs.washington.edu/~pedrod/papers/kdd00.pdf)
* [Mining Time-changing Data Streams, KDD 2001](https://www.researchgate.net/publication/2375511_Mining_Time-Changing_Data_Streams)
* [Efficient Decision Tree Construction on Streaming Data, KDD 2003](http://web.cse.ohio-state.edu/~agrawal.28/p/sigkdd03.pdf)
* [Incremental Learning with Partial Instance Memory, Artificial Intelligence 2004](https://www.sciencedirect.com/science/article/pii/S0004370203001498)
* [Data Streams Classification by Incremental Rule Learning with Parameterized Generalization, SAC 2006](https://www.researchgate.net/publication/221001229_Data_streams_classification_by_incremental_rule_learning_with_parameterized_generalization)
* [Rudolf: Interactive Rule Refinement System for Fraud Detection, VLDB 2016](http://www.vldb.org/pvldb/vol9/p1465-milo.pdf)
* [Rule Sharing for Fraud Detection via Adaptation, ICDE 2018](https://slavanov.com/research/icde18.pdf)
* [GOLDRUSH: Rule Sharing System for Fraud Detection, VLDB 2018](http://www.vldb.org/pvldb/vol11/p1998-jarovsky.pdf)

Incremental supervised methods assume that labels become available together with the input (or at least with a delay that is tolerable). When labels are scarse or not available at all, unsupervised or semisupervised incremental methods are the only solution. To the best of my knowledge, there are no rule-based methods that can work in these settings.


## Time-series classification/forecasting with neural networks + attention
Attention mechanisms have recently gained popularity in training neural networks. They have shown their potential for machine translations, overcoming most of the limitations of standard recurrent neural networks, and allowing for more interpretable models. Recently, there have been several attempts to extend the attention mechanism to (multi-variate) time-series.

* [Neural Machine Translation by Jointly Learning to Align and Translate, ICLR 2015](https://arxiv.org/abs/1409.0473)
* [Effective Approaches to Attention-based Neural Machine Translation, EMNLP 2015](https://arxiv.org/abs/1508.04025)
* [RETAIN: An Interpretable Predictive Model for Healthcare using Reverse Time Attention Mechanism, NIPS 2016](https://arxiv.org/abs/1608.05745)
* [A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction, IJCAI 2017](https://arxiv.org/abs/1704.02971)
* [Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks, SIGIR 2018](https://arxiv.org/abs/1703.07015)
* [Attend and Diagnose: Clinical Time Series Analysis using Attention Models, AAAI 2018](https://arxiv.org/abs/1711.03905)
* [Temporal Pattern Attention for Multivariate Time Series Forecasting, ECML 2019](https://arxiv.org/abs/1809.04206)
* [CDSA: Cross-Dimensional Self-Attention for Multivariate, Geo-tagged Time Series Imputation, NIPS 2019](https://arxiv.org/abs/1905.09904)
* [Exploring Interpretable LSTM Neural Networks over Multi-Variable Data, ICML 2019](https://arxiv.org/abs/1905.12034)
* [Enhancing the Locality and Breaking the Memory Bottleneck of Transformer on Time Series Forecasting, NIPS 2019](https://arxiv.org/abs/1907.00235)

## Others
* [Detecting Bias in Black-Box Models Using Transparent Model Distillation](https://arxiv.org/pdf/1710.06169.pdf)
* [Deep Learning for Case-Based Reasoning through Prototypes: A Neural Network that Explains Its Predictions](https://arxiv.org/pdf/1710.04806.pdf)
* [Explainable Artificial Intelligence: Understanding, Visualizing and Interpreting Deep Learning Models](arxiv.org/pdf/1708.08296.pdf)
* [Visual Interpretability for Deep Learning: a Survey](https://arxiv.org/pdf/1802.00614.pdf)
* [A Survey Of Methods For Explaining Black Box Models](arxiv.org/pdf/1802.01933.pdf)
* [Manipulating and Measuring Model Interpretability](arxiv.org/pdf/1802.07810.pdf)
* [Interpretation of Neural Networks is Fragile](arxiv.org/pdf/1710.10547.pdf)
* [Interpretation of Prediction Models Using the Input Gradient](arxiv.org/pdf/1611.07634.pdf)
* [Programs as Black-Box Explanations](arxiv.org/pdf/1611.07579.pdf)
* ["I know it when I see it". Visualization and Intuitive Interpretability](arxiv.org/pdf/1711.08042.pdf)
* ["Why Should I Trust You?": Explaining the Predictions of Any Classifier](arxiv.org/pdf/1602.04938.pdf)
* [Beyond Sparsity: Tree Regularization of Deep Models for Interpretability](arxiv.org/pdf/1711.06178.pdf)
* [European Union regulations on algorithmic decision-making and a "right to explanation"](arxiv.org/pdf/1606.08813.pdf)
* [Explanation in Artificial Intelligence: Insights from the Social Sciences](arxiv.org/abs/1706.07269)
* [Extracting Tree-Structured Representations of Trained Networks](papers.nips.cc/paper/1152-extracting-tree-structured-representations-of-trained-networks.pdf)
* [Interpretability of Deep Learning Models: A Survey of Results](orca.cf.ac.uk/101500/1/Interpretability%20of%20Deep%20Learning%20Models%20-%20A%20Survey%20of%20Results.pdf)
* [Interpretable and Pedagogical Examples](arxiv.org/pdf/1711.00694.pdf)
* [Inverse Classification for Comparison-based Interpretability in Machine Learning](https://arxiv.org/pdf/1712.08443.pdf)
* [Peeking Inside the Black Box: Visualizing Statistical Learning with Plots of Individual Conditional Expectation](arxiv.org/pdf/1309.6392.pdf)
* [Show, Attend, Control, and Justify: Interpretable Learning for Self-Driving Cars](kimjinkyu.files.wordpress.com/2017/12/nips_2017.pdf)
* [The Bayesian Case Model: A Generative Approach for Case-Based Reasoning and Prototype Classification](arxiv.org/pdf/1503.01161.pdf)
* [The Doctor Just Won't Accept That!](arxiv.org/pdf/1711.08037.pdf)
* [The Intriguing Properties of Model Explanations](arxiv.org/pdf/1801.09808.pdf)
* [The Mythos of Model Interpretability](arxiv.org/pdf/1606.03490.pdf)
* [The Promise and Peril of Human Evaluation for Model Interpretability](arxiv.org/pdf/1711.07414.pdf)
* [Towards A Rigorous Science of Interpretable Machine Learning](arxiv.org/pdf/1702.08608.pdf)
* [TreeView: Peeking into Deep Neural Networks Via Feature-Space Partitioning](arxiv.org/pdf/1611.07429.pdf)
* [Using Visual Analytics to Interpret Predictive Machine Learning Models](arxiv.org/pdf/1606.05685.pdf)
* [Contextual Explanation Networks](arxiv.org/pdf/1705.10301.pdf)

### To-read
* [A Model Explanation System: Latest Updates and Extensions](https://arxiv.org/pdf/1606.09517.pdf)
* [Explaining Classification Models Built on High-Dimensional Sparse Data](arxiv.org/pdf/1607.06280.pdf)
* [How to Explain Individual Classification Decisions](https://arxiv.org/pdf/0912.1128.pdf)
* [MAGIX: Model Agnostic Globally Interpretable Explanations](https://arxiv.org/pdf/1706.07160.pdf)

## Rule Extraction / Rule mining
* [Survey and critique of techniques for extracting rules from trained artificial neural networks](https://www.sciencedirect.com/science/article/pii/0950705196819204)
* [The truth will come to light: Directions and challenges in extracting the knowledge embedded within trained artificial neural networks](http://ieeexplore.ieee.org/document/728352/)
* [Using Sampling and Queries to Extract Rules from Trained Neural Networks](https://www.sciencedirect.com/science/article/pii/B9781558603356500131)
* [Active Learning-Based Pedagogical Rule Extraction](http://ieeexplore.ieee.org/document/7018925/)
* [A Statistics based Approach for Extracting Priority Rules from Trained Neural Networks](http://ieeexplore.ieee.org/document/861337/)
* [Extracting Tree-Structured Representations of Trained Networks](papers.nips.cc/paper/1152-extracting-tree-structured-representations-of-trained-networks.pdf)
* [Extraction of Fuzzy Rules from Trained Neural Network Using Evolutionary Algorithm](https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2003-9.pdf)
* [A Unified Approach to the Extraction of Rules from Artificial Neural Networks and Support Vector Machines](http://home.iscte-iul.pt/~dmt/publ/2010_A_unified_approach_to_extraction_of_rules.pdf)
* [Extracting Provably Correct Rules from Artificial Neural Networks](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.2.2110&rep=rep1&type=pdf)
* [Orthogonal Search-based Rule Extraction (OSRE) for Trained Neural Networks: A Practical and Efficient Approach](http://ieeexplore.ieee.org/document/1603623/)
* [Three Techniques For Extracting Rules from Feedforward Networks](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.50.1175&rep=rep1&type=pdf)
* [Extracting Provably Correct Rules from Artificial Neural Networks](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.2.2110&rep=rep1&type=pdf)
* [Learning Explanatory Rules From Noisy Data](https://arxiv.org/abs/1711.04574)
* [Machine-Learning Based Circuit Synthesis](https://ieeexplore.ieee.org/document/6377134/)
* [Synthesizing Entity Matching Rules by Examples](https://www.researchgate.net/publication/321767065_Synthesizing_entity_matching_rules_by_examples)
* [Discovering Denial Constraints](http://www.vldb.org/pvldb/vol6/p1498-papotti.pdf)
* [Switching Neural Networks: A New Connectionist Model for Classification](http://www.rulex.ai/wp-content/uploads/2017/04/Switching-Neural-Networks-A-New-Connectionist-Model-for-Classification.pdf)
* [Shadow Clustering: A Method for Monotone Boolean Function Synthesis](https://pdfs.semanticscholar.org/2cee/d0be359ad6e91f856f42ba63f1d5866e1a20.pdf)

### To-read
* [Learning Accurate And Understandable Rules form SVM Classifiers](https://pdfs.semanticscholar.org/ec7c/0ff68dbe73ed2ff1944b53070b223a371c25.pdf)
* [ITER: An Algorithm for Predictive Regression Rule Extraction](https://link.springer.com/chapter/10.1007/11823728_26)
* [Rule Extraction from Training Data Using Neural Network](https://www.researchgate.net/profile/Manomita_Chakraborty/publication/312086580_Rule_Extraction_from_Training_Data_Using_Neural_Network/links/59e598d945851525024e223e/Rule-Extraction-from-Training-Data-Using-Neural-Network.pdf)
* [Reverse engineering the neural networks for rule extraction in classification problems](https://link.springer.com/article/10.1007/s11063-011-9207-8)
* [Rule-Neg: Extracting Rules from Trained Ann](https://www.amazon.co.uk/Rule-Neg-Extracting-Rules-Trained-Ann/dp/B001A2ISO2)
* [X-TREPAN: A multi class regression and adapted extraction of comprehensible decision tree in artificial neural networks](https://arxiv.org/ftp/arxiv/papers/1508/1508.07551.pdf)
