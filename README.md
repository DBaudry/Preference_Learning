# Preference Learning with Gaussian Processes 

This project is based on the article *Preference Learning with Gaussian Processes* (Wei Chu and Zoubin Ghahramani, 2005) 
(link [here](http://mlg.eng.cam.ac.uk/zoubin/papers/icml05chuwei-pl.pdf)).

The main contribution of this artice was to propose a full non-parametric approach to tackle problems such as : Instance Ranking,
Label classification and Label ranking. We propose here an implementation of the algorithms in the paper, different algorithms to
compare their performance on the learning task, and finally an application to solve the widely studied Cold-Start 
problem in a movie recommender system.
 
Our code is divided into 4 types of files :
* files for Instance Preference Learning : *Instance_learning.py*, *SVM_IL.py* and *main_IL.py* 
* files for Label Preference Learning : *Label_learning.py*, *SVM_LL.py* and *main_LL.py*
* *movie_suggestion.py* for the implementation of the movie recommender system and the related experiments
* "Utils": *input_data.py* to download and pre-process the real datasets or sample toy datasets, *utils.py* to store some
functions that are useful in several files, and finally *expe.py* to define the procedures for the experiments in 
order to make the main files easier to understand.

So there are three ways to directly test our algorithms :
* Run *main_IL.py* to test Instance Preference Learning
* Run *main_LL.py* to test Label Preference Learning
* Run *movie_suggestion.py* to test the experiment with the movie recommender

We provide below a more detailed description of the files containing the algorithms for each of the three problems.

## Instance Preference Learning

We define the problem as in the original article: different instances (represented as features vectors of size *d*) are available. We 
want to know if an agent prefers an instance *a* over an instance *b*, but the only information we have about him is a set of *m*
observed preferences. 

Our class *learning_instance_preference* takes as input the design matrix of the instances and the set of observed preference, and follows the steps desrcibed in the article to compute the *maximum a posteriori* of the function *f* over the available instances.
Then this result is used to predict the value of *f* and perform pairwise comparison for any other set of instances.
To compare this algorithm we also implemented two benchmark algorithms. These are adaptations from articles from Herbrich et al. and Har-Peled et al. (constraint classification). They mainly rely on the translation of preferences into vectors such that one can handle the problem as a classification problem. It is performed inside our classes *SVM_InstancePref* and *CCSVM_IL*.

Results are tested using a simple accuracy score : we compare random pairs of instances and check the percentage of pairs that 
are correctly predicted (instances can be drawn from the train set or any test set depending on what we want to check).

## Label Preference Learning

This problem is slightly different from the previous one : now there are multiple agents, and for each of them an incomplete preference 
graph on the same *L* different instances. The idea here is to complete the preference graph for each agent, and to be able to predict the
preference graph of any new agent. We may expect two different kinds of performance from our algorithm :
* *Classification problem*: we want to know which label will be prefered among the others, so we only need to correctly recover the *L-1*
edges involving this label in the preference graph. Accuracy will be based on the ability to find this best label for each agent.
* *Ranking problem*: we want the complete preference graph for each user.

The class *learning_label_preference* takes as inputs the design matrix of the agent features and the list of the observed preference graphs
 (a preference graph being simply represented as the list of the preferences that form the edges). Considering this, the code is organized 
exactly as the one for instance learning since the two algorithms are very similar.
Again, the constraint-classification SVM algorithm from Har-Peled et al. is used to benchmark this algorithm. The implementation of this algorithm is done in our class *CCSVM_LL*.

The classification problem is again tested using the accuracy. For the ranking problem, we consider that the algorithm is
 wrong if the observed graph for a given agent is not consistent with the prediction (i.e if any edge is in the wrong direction).

## Movie recommender system

For this part we used the MovieLens database to build a user-movie matrix M where M(i,j) is the rating of the movie j given by user i.
Then a low-rank matrix factorization has been used to decompose this matrix in $M=UV^T$ (explained in the guidelines [here](https://github.com/DBaudry/Preference_Learning/blob/master/Articles/homework%20Linear%20UCB%20MVA.pdf)), where U is a
matrix containing user's features and V a matrix containing movie's features. So the rating M(i,j) is just the scalar product between u(i)
 and v(j).

We considered the cold start problem where a new user joins the platform and we need to learn his preferences. We used the Instance
Preference Learning for this problem : we do not observe u but we have the complete instance matrix V, and we solve this problem sequentially as:
* A movie *a* is suggested to the user
* The user watches this movie and gives a (noisy) rating r(a, t)
* This rating is compared with the ratings of previously watched movies to update the preferences of the user
* The *learning_instance_preference* class is called to compute the new values of *f* for the already watched movies
* Using the same class, prediction is performed on the movies left to watch

New movies are added at each step from the original pool to keep the same size for the set of instances. To avoid too long computation time
we stop the update of *f* after an adaptation period. To compare our results with other algorithms, we considered a solution for this problem
using the Linear Bandit Framework (again from the guidelines [here](https://github.com/DBaudry/Preference_Learning/blob/master/Articles/homework%20Linear%20UCB%20MVA.pdf)).

For this problem both implementation and experiments are available in *movie_suggestion.py*.
