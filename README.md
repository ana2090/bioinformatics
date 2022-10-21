**Bioinformatics Projects**

1. k-Nearest Neighbors: An implementation of the k-Nearest Neighbors algorithm on two datasets.
  It works by going through each sample in the dataset, using some distance metric to measure the
  distances between the current sample and every other sample, and choosing the nearest k of them. Then it
  checks the labels of those chosen samples and checks them against the real label of the current sample.
  
  Firstly, I normalize the dataset by taking the mean and standard deviation of each feature, then subtracting
  the means from their corresponding samples and dividing that by the standard deviation:
    𝑋𝑖𝑗 = [𝑋𝑖𝑗 − 𝑋𝑗.𝑚𝑒𝑎𝑛()] / 𝑋𝑗.𝑠𝑡𝑑()
  This normalized matrix then gets passed to the kNN algorithm along with its corresponding labels and an indication
  of what type of distance algorithm to use. The distance metrics are cosine, euclidean, and manhattan city block.
  The scipy package was used for these distance functions. The kNN function itself iterates through each sample in X,
  uses the distance metric to get the k nearest samples, tallies the labels that each of those samples have, then if at
  least half have the same label as the current sample, the prediction is marked as correct. The accuracy is returned.
  
2. Linear Regression: I made a linear classifier in three different ways. Firstly using the closed-form solution, where
  you can directly solve for the weights of the linear combination. Secondly using a linear learning algorithm, where you
  fit the model by reading samples from the feature matrix one by one and updating the weights as you go based on the
  correctness of the prediction using the existing weights. The last classifier is a logistic regression classifier that
  uses gradient descent to reach the optimal weights.

3. Skews: I used frequency analysis and pattern matching to calculate the skews of k-mers in the genome sequence of e.coli
  in order to deduce what segment of the genome the replication origin is in. In this case, the segment is just a smaller
  section of the genetic sequence. The general steps to solve the problem were to first break the genome sequence to some
  amount of segments of equal length. Next I found all the k-mers in each of those segments, then counted the occurrences
  of each of the k-mers within each segment. Lastly I found the skews of each of the k-mers. I created a basic table of the
  skews and then graphed them. One area of improvement in this project is making it more efficent in terms of computation 
  time and storage.
