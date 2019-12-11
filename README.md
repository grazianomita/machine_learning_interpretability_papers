# Model Interpretability
This is a public collection of papers related to machine learning (ML) model interpretability. Someone might think that ML interpretability is a recent field: nothing more wrong! Throughout the history of machine learning, researchers have always proposed methods that are interpretable or can "explain things in human-understandable terms"! Unfortunately it is very hard to make an exhaustive list of works because many papers, especially the oldest ones, had not the word "interpretable/interpretability/explainability" in the title. Thus, the works reported here are just a small subset of what has been done in the field (and I personally read), and refers to sub-areas of model interpretability.


## Introduction to Model Interpretability
These are some of the more recent references to read if you are approaching the field of model interpretability and you want to have a general view of the field. "Explaining things" is not a machine learning specific concept: there is tons of research in philosphy, psychology, sociology, of how people generate/evaluate explanations. Restricting our attention to machine learning, interpreting a model might mean understanding how the model works interally; it might also translate into explaining why a model answers in a certain way! These are examples of different "types" of explanations. The way we explain things is often model and application specific! Last but not least, we should also be able to evaluate and compare the quality of explanations.

* [European Union regulations on algorithmic decision-making and a "right to explanation", AI Magazine 2016](https://arxiv.org/abs/1606.08813)
* [“Why Should I Trust You?” Explaining the Predictions of Any Classifier, KDD 2016](https://arxiv.org/abs/1602.04938)
* [The Mythos of Model Interpretability, ICML 2016](https://arxiv.org/abs/1606.03490)
* [Model-Agnostic Interpretability of Machine Learning, ICML 2016](https://arxiv.org/abs/1606.05386)
* [Towards A Rigorous Science of Interpretable Machine Learning, 2017](https://arxiv.org/abs/1702.08608)
* [Transparency: Motivations and Challenges, ICML 2017](https://arxiv.org/abs/1708.01870)
* [Explanation in Artificial Intelligence: Insights from the Social Sciences, Artificial Intelligence 2017](https://arxiv.org/abs/1706.07269)
* [The intriguing Properties of Model Explanation, NIPS 2017](https://arxiv.org/abs/1801.09808)
* [The Doctor Just Won’t Accept That!, NIPS 2017](https://arxiv.org/abs/1711.08037)
* ["I know it when I see it". Visualization and Intuitive Interpretability, NIPS 2017](https://arxiv.org/abs/1711.08042)
* [The Promise and Peril of Human Evaluation for Model Interpretability, NIPS 2017](https://arxiv.org/abs/1711.07414)
* [Interpretability of deep learning models: a survey of results, DAIS 2017](https://ieeexplore.ieee.org/document/8397411)
* [A Survey Of Methods For Explaining Black Box Models, ACM Computing Surveys 2018](https://arxiv.org/abs/1802.01933)
* [Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and use Interpretable Models Instead, Nature Machine Learning 2019](https://arxiv.org/abs/1811.10154)
* [Interpretable Machine Learning - A Guide for Making Black Box Models Explainable, 2019](https://christophm.github.io/interpretable-ml-book/)


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

* [Learning Decision Lists, Machine Learning 1987](https://people.csail.mit.edu/rivest/pubs/Riv87b.pdf)
* [The CN2 Induction Algorithm, Machine Learning 1989](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.23.736&rep=rep1&type=pdf)
* [Very simple classification rules perform well on most commonly used datasets, Machine Learning 1993](https://www.mlpack.org/papers/ds.pdf)
* [FOIL: A midterm report, ECML 1993](https://link.springer.com/chapter/10.1007/3-540-56602-3_124)
* [Incremental reduced error pruning, ICML 1994](https://www.researchgate.net/publication/2271856_Incremental_Reduced_Error_Pruning)
* [Fast Effective Rule Induction, ICML 1995](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.107.2612&rep=rep1&type=pdf)
* [Unifying instance-based and rule-based induction, Machine Learning 1996](https://link.springer.com/article/10.1007/BF00058656)
* [A Simple, Fast, and Effective Rule Learner, AAAI 1999](https://pdfs.semanticscholar.org/7237/70d9ac418e923db5e087ae18c04702f5986e.pdf)
* [The Set Covering Machine, JMLR 2002](http://www.jmlr.org/papers/volume3/marchand02a/marchand02a.pdf)
* [Binary Rule Generation via Hamming Clustering, IEEE TKDE 2002](https://www.semanticscholar.org/paper/Binary-Rule-Generation-via-Hamming-Clustering-Muselli-Liberati/3b01a190d4e55929273caf5cb272b69e0c055ff2)
* [Disjunctions of Conjunctions, Cognitive Simplicity, and Consideration Sets, Journal of Marketing Research 2010](https://pdfs.semanticscholar.org/bf65/dc3164be43919088695d7f43ee2e51d4b614.pdf)
* [Finding a Short and Accurate Decision Rule in Disjunctive Normal Form by Exhaustive Search, Machine Learning 2010](https://link.springer.com/article/10.1007/s10994-010-5168-9)
* [BRACID: a comprehensive approach to learning rules from imbalanced data, Journal of Intelligent Information Systems 2012](https://link.springer.com/article/10.1007/s10844-011-0193-0)
* [An Integer Optimization Approach to Associative Classification, NIPS 2012](https://pdfs.semanticscholar.org/ec2c/b3a9abdcad58ea4acb73b1411f7eb98d8472.pdf)
* [Exact Rule Learning via Boolean Compressed Sensing, ICML 2013](http://ssg.mit.edu/~krv/pubs/MalioutovV_icml2013.pdf)
* [Foundations of Rule Learning, 2014](https://www.springer.com/gp/book/9783540751960)
* [Box Drawings for Learning with Imbalanced Data, KDD 2014](https://arxiv.org/abs/1403.3378)
* [Falling Rule Lists, AISTATS 2015](https://arxiv.org/abs/1411.5899)
* [Bayesian Or’s of And’s for Interpretable Classification with Application to Context Aware Recommender Systems, 2015](https://finale.seas.harvard.edu/files/finale/files/techreportboa_wangetal.pdf)
* [nterpretable classifiers using rules and Bayesian analysis: Building a better stroke prediction model, Annals of Applied Statistics 2015](https://arxiv.org/abs/1511.01644)
* [Interpretable Decision Sets: A Joint Framework for Description and Prediction, KDD 2016](https://www-cs-faculty.stanford.edu/people/jure/pubs/interpretable-kdd16.pdf)
* [Interpretable Two-level Boolean Rule Learning for Classification, ICML 2016](https://arxiv.org/abs/1606.05798)
* [Scalable Bayesian Rule Lists, ICML 2017](https://arxiv.org/abs/1602.08610)
* [A Bayesian Framework for Learning Rule Sets for Interpretable Classification, JMLR 2017](http://jmlr.org/papers/v18/16-003.html)
* [Learning Credible Models, KDD 2018](https://arxiv.org/abs/1711.03190)
* [Learning Certifiably Optimal Rule Lists for Categorical Data, JMLR 2018](https://arxiv.org/abs/1704.01701)
* [An Optimization Approach to Learning Falling Rule Lists, AISTATS 2018](http://proceedings.mlr.press/v84/chen18a/chen18a.pdf)
* [Boolean Decision Rules via Column Generation, NIPS 2018](https://arxiv.org/abs/1805.09901)
* [An Interpretable Model with Globally Consistent Explanations for Credit Risk, NIPS 2018](https://arxiv.org/abs/1811.12615)


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


## Case-Based Interpretable Models
Case-based models like KNN (k-nearest neighbors) classify new records based on previously seen records. More specifically, every new input is compared with previously seen records (usually its neighbors) and classified according to a measure of distance. For case-based models, interpretability translates into similarity: a given input is assigned to class C if it is similar to other samples that were previously classified in the same way. I am not going to report here all the well-known work on case-based methods. Instead, I will focus on recent papers that I think are innovative and interesting to read.

* [The Bayesian Case Model: A Generative Approach for Case-Based Reasoning and Prototype Classification, NIPS 2014](https://users.cs.duke.edu/~cynthia/docs/KimRuSh14.pdf)
* [Bayesian Patchworks: An Approach to Case-Based Reasoning, Machine Learning 2018](https://arxiv.org/abs/1809.03541)
* [Deep Learning for Case-based Reasoning through Prototypes: A Neural Network that Explains its Predictions, AAAI 2018](https://arxiv.org/abs/1710.04806)
* [This Looks Like That: Deep Learning for Interpretable Image Recognition, NIPS 2019](https://arxiv.org/abs/1806.10574)


## Post-hoc interpretability of Artificial Neural Networks
Artificial neural networks are the black-box model "par excellence". Thus, it is not surprising that a significant part of the publications on model interpretability is neural network specific. Unfortunately, I have not spent enough time on this specific sub-field, but I hope these short list of papers might be useful.

* [Extracting Provably Correct Rules from Artificial Neural Networks, Technical Report 1993](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.2.2110&rep=rep1&type=pdf)
* [Extracting refined rules from knowledge-based neural networks, Machine Learning 1993](https://link.springer.com/article/10.1007/BF00993103)
* [Using Sampling and Queries to Extract Rules from Trained Neural Networks, ICML 1994](https://www.semanticscholar.org/paper/Using-Sampling-and-Queries-to-Extract-Rules-from-Craven-Shavlik/ba176454d1ade52e6eec74e3f9eed5f61179761a)
* [Rule-Neg: Extracting Rules from Trained Ann, 1994](https://www.amazon.co.uk/Rule-Neg-Extracting-Rules-Trained-Ann/dp/B001A2ISO2)
* [Survey and critique of techniques for extracting rules from trained artificial neural networks, KBS 1995](https://www.sciencedirect.com/science/article/pii/0950705196819204)
* [Extracting Rules from Artificial Neural Networks with Distributed Representations, NIPS 1995](https://papers.nips.cc/paper/924-extracting-rules-from-artificial-neural-networks-with-distributed-representations.pdf)
* [Extracting Tree-Structured Representations of Trained Networks, NIPS 1996](https://papers.nips.cc/paper/1152-extracting-tree-structured-representations-of-trained-networks.pdf)
* [The truth will come to light: directions and challenges in extracting the knowledge embedded within trained artificial neural networks, IEEE TNN 1998](https://ieeexplore.ieee.org/document/728352)
* [A statistics based approach for extracting priority rules from trained neural networks, IEEE IJCNN 2000](https://ieeexplore.ieee.org/document/861337)
* [The truth is in there : directions and challenges in extracting rules from trained ar tificial neural networks, IEEE TNN 2000](https://www.researchgate.net/publication/2614662_The_Truth_is_in_There_Directions_and_Challenges_in_Extracting_Rules_From_Trained_Artificial_Neural_Networks)
* [Extracting Rules from Trained Neural Networks, IEEE TNN 2000](https://ieeexplore.ieee.org/document/839008)
* [Interpretation of Trained Neural Networks by Rule Extraction, ICCI 2001](https://link.springer.com/chapter/10.1007/3-540-45493-4_20)
* [A new methodology of extraction, optimization and application of crisp and fuzzy logical rules, IEEE TNN 2001](https://www.semanticscholar.org/paper/A-new-methodology-of-extraction%2C-optimization-and-Duch-Adamczak/6538e4f542850fcf8c1e9e413546f1616ed79850)
* [Diagnostic Rule Extraction from Trained Feedforward Neural Networks, Mechanical Systems and Signal Processing 2002](https://www.sciencedirect.com/science/article/pii/S0888327001913962)
* [Extraction of Fuzzy Rules from Trained Neural Network Using Evolutionary Algorithm, ESAN 2003](https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2003-9.pdf)
* [Are artificial neural networks white boxes, IEEE TNN 2005](https://www.researchgate.net/publication/7638457_Are_Artificial_Neural_Networks_White_Boxes)
* [Orthogonal search-based rule extraction (OSRE) for trained neural networks: a practical and efficient approach, IEEE TNN 2006](https://ieeexplore.ieee.org/document/1603623)
* [A modified fuzzy min-max neural network with rule extraction, Applied Soft Computing 2008](https://www.sciencedirect.com/science/article/pii/S1568494607000865)
* [Reverse engineering the neural networks for rule extraction in classification problems, Neural Processing Letters 2012](https://link.springer.com/article/10.1007/s11063-011-9207-8)
* [Active Learning-Based Pedagogical Rule Extraction, IEEE TNN 2015](https://ieeexplore.ieee.org/document/7018925)
* [TreeView: Peeking into Deep Neural Networks Via Feature-Space Partitioning, NIPS 2016](https://arxiv.org/abs/1611.07429)
* [Interpretation of Neural Networks is Fragile, AAAI 2019](arxiv.org/pdf/1710.10547.pdf)


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


## Unsupervised Learning of Disentangled Representations

Representation Learning consists of learning representations of the data that make model classification/prediction easier.  In particular, there has been a growing interest for learning interpretable representations that can be understood to humans, more specifically disentangled representations. A representation is said to be disentangled when "a change in one dimension corresponds to a change in one factor of variation, while being relatively invariant to changes in other factors". When we use generative models to learn disentangled representations, the goal is to learn a latent space such that latent dimensions (or a subset of them) encode independent factors of variations (that are supposed to be involved in the generation of the data). This should be done in an unsupervised way, that is without explicitly knowing which are the generative factors of variations at training time.

* [Learning factorial codes by predictability minimization, Neural Computation 1992](https://ieeexplore.ieee.org/document/6795705)
* [Disentangling factors of variation via generative entangling, 2012](https://arxiv.org/abs/1210.5474)
* [Representation learning: A review and new perspectives, IEEE TSE 2013](https://arxiv.org/abs/1206.5538)
* [Tensor analyzers, ICML 2013](http://proceedings.mlr.press/v28/tang13.html)
* [Learning the irreducible representations of commutative lie groups, 2014](https://arxiv.org/abs/1402.4437)
* [Transformation properties of learned visual representations, ICLR 2015](https://arxiv.org/abs/1412.7659)
* [Adversarial Autoencoders, ICLR 2016](https://arxiv.org/abs/1511.05644)
* [Information Dropout: Learning Optimal Representations Through Noisy Computation, IEEE TPAMI 2016](https://arxiv.org/abs/1611.01353)
* [Infogan: Interpretable representation learning by information maximizing generative adversarial nets, NIPS 2016](https://arxiv.org/abs/1606.03657)
* [β-VAE: Learning basic visual concepts with a constrained variational framework, ICLR 2017](https://openreview.net/forum?id=Sy2fzU9gl)
* [Understanding disentangling in β-VAE, NIPS 2017](https://arxiv.org/abs/1804.03599)
* [Variational Inference of Disentangled Latent Concepts from Unlabeled Observations, ICLR 2018](https://arxiv.org/abs/1711.00848)
* [Disentangling by Factorising, ICML 2018](https://arxiv.org/abs/1802.05983)
* [Disentangled Sequential Autoencoder, ICML 2018](https://arxiv.org/abs/1803.02991)
* [On the emergence of invariance and disentangling in deep representations, JMLR 2018](https://arxiv.org/abs/1706.01350)
* [Learning Disentangled Joint Continuous and Discrete Representations, NIPS 2018](https://arxiv.org/abs/1804.00104)
* [Information Constraints on Auto-Encoding Variational Bayes, NIPS 2018](https://arxiv.org/abs/1805.08672)
* [Isolating Sources of Disentanglement in VAEs, NIPS 2018](https://arxiv.org/pdf/1802.04942.pdf)
* [Learning deep disentangled embeddings with the f-statistic loss, NIPS 2018](https://arxiv.org/abs/1802.05312)
* [InfoVAE: Balancing Learning and Inference in Variational Autoencoders, AAAI 2019](https://arxiv.org/abs/1706.02262)
* [Auto-Encoding Total Correlation Explanation, AISTATS 2019](https://arxiv.org/abs/1802.05822)
* [Structured Disentangled Representations, AISTATS 2019](https://arxiv.org/abs/1804.02086)
* [Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations, ICML 2019](https://arxiv.org/abs/1811.12359)
* [Robustly Disentangled Causal Mechanisms: Validating Deep Representations for Interventional Robustness, ICML 2019](https://arxiv.org/abs/1811.00007)


## Weakly-supervised Learning of Disentangled Representations

A recent result from Locatello et al. 2019, showed that unsupervised disentanglement learning is fundamentally impossible. That is why recent research is focusing on weakly-supervised methods where we do not need to explicitly provide the generative factors at training time, but we still can provide some kind of supervision. An example of weak supervision can for example be the class label. Alternatively, we could also group input records together if we think they share some generative factors. Here, I report the most recent works on the field.

* [Semi-Supervised Learning with Deep Generative Models, NIPS 2014](https://arxiv.org/abs/1406.5298)
* [Learning Disentangled Representations with Semi-Supervised Deep Generative Models, NIPS 2017](https://arxiv.org/abs/1706.00400)
* [Disentangling Factors of Variation with Cycle-Consistent Variational Auto-Encoders, ECCV 2018](https://arxiv.org/abs/1804.10469)
* [Multi-level variational autoencoder: Learning disentangled representations from grouped observations, AAAI 2018](https://arxiv.org/abs/1705.08841)
* [Dual Swap Disentangling, NIPS 2018](https://arxiv.org/abs/1805.10583)
* [Gaussian Process Prior Variational Autoencoders, NIPS 2018](https://arxiv.org/abs/1810.11738)
* [Weakly Supervised Disentanglement with Guarantees, under review to ICLR 2020](https://arxiv.org/abs/1910.09772)
* [Demystifying Inter-Class Disentanglement, under review to ICLR 2020](https://arxiv.org/abs/1906.11796)
* [Weakly Supervised Disentanglement by Pairwise Similarities, under review to ? 2020](https://arxiv.org/abs/1906.01044)


## Others
Here you can find all the papers that do not fit in the previous sections. I am going to reorganize them soon or later.

* [Detecting Bias in Black-Box Models Using Transparent Model Distillation](https://arxiv.org/pdf/1710.06169.pdf)
* [Deep Learning for Case-Based Reasoning through Prototypes: A Neural Network that Explains Its Predictions](https://arxiv.org/pdf/1710.04806.pdf)
* [Explainable Artificial Intelligence: Understanding, Visualizing and Interpreting Deep Learning Models](arxiv.org/pdf/1708.08296.pdf)
* [Visual Interpretability for Deep Learning: a Survey](https://arxiv.org/pdf/1802.00614.pdf)
* [A Survey Of Methods For Explaining Black Box Models](arxiv.org/pdf/1802.01933.pdf)
* [Manipulating and Measuring Model Interpretability](arxiv.org/pdf/1802.07810.pdf)
* [Interpretation of Prediction Models Using the Input Gradient](arxiv.org/pdf/1611.07634.pdf)
* [Programs as Black-Box Explanations](arxiv.org/pdf/1611.07579.pdf)
* [Beyond Sparsity: Tree Regularization of Deep Models for Interpretability](arxiv.org/pdf/1711.06178.pdf)
* [Interpretable and Pedagogical Examples](arxiv.org/pdf/1711.00694.pdf)
* [Inverse Classification for Comparison-based Interpretability in Machine Learning](https://arxiv.org/pdf/1712.08443.pdf)
* [Peeking Inside the Black Box: Visualizing Statistical Learning with Plots of Individual Conditional Expectation](arxiv.org/pdf/1309.6392.pdf)
* [Show, Attend, Control, and Justify: Interpretable Learning for Self-Driving Cars](kimjinkyu.files.wordpress.com/2017/12/nips_2017.pdf)
* [Using Visual Analytics to Interpret Predictive Machine Learning Models](arxiv.org/pdf/1606.05685.pdf)
* [Contextual Explanation Networks](arxiv.org/pdf/1705.10301.pdf)
* [A Model Explanation System: Latest Updates and Extensions](https://arxiv.org/pdf/1606.09517.pdf)
* [Explaining Classification Models Built on High-Dimensional Sparse Data](arxiv.org/pdf/1607.06280.pdf)
* [How to Explain Individual Classification Decisions](https://arxiv.org/pdf/0912.1128.pdf)
* [MAGIX: Model Agnostic Globally Interpretable Explanations](https://arxiv.org/pdf/1706.07160.pdf)
* [Learning Accurate And Understandable Rules form SVM Classifiers](https://pdfs.semanticscholar.org/ec7c/0ff68dbe73ed2ff1944b53070b223a371c25.pdf)
* [X-TREPAN: A multi class regression and adapted extraction of comprehensible decision tree in artificial neural networks](https://arxiv.org/ftp/arxiv/papers/1508/1508.07551.pdf)
* [Learning Explanatory Rules From Noisy Data](https://arxiv.org/abs/1711.04574)
* [Machine-Learning Based Circuit Synthesis](https://ieeexplore.ieee.org/document/6377134/)
* [Synthesizing Entity Matching Rules by Examples](https://www.researchgate.net/publication/321767065_Synthesizing_entity_matching_rules_by_examples)
* [Discovering Denial Constraints](http://www.vldb.org/pvldb/vol6/p1498-papotti.pdf)
* [Switching Neural Networks: A New Connectionist Model for Classification](http://www.rulex.ai/wp-content/uploads/2017/04/Switching-Neural-Networks-A-New-Connectionist-Model-for-Classification.pdf)
* [Shadow Clustering: A Method for Monotone Boolean Function Synthesis](https://pdfs.semanticscholar.org/2cee/d0be359ad6e91f856f42ba63f1d5866e1a20.pdf)
