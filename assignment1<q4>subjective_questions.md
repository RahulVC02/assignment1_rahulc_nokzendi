<h3>Theoretical time complexity:</h3>


The time complexity depends on the output size and the number of attributes in the input data table. Additionally, the input type (continuous or discrete) can also affect the complexity. The following are the variables used- 
1. Output size or the number of data points- $N$
2. Number of attributes or columns in the input - $M$
3. The number of unique values in the $i^{th}$ column in the input table - $d_i$

Now, going through the different cases: 

<h4>1. Real input values: </h4>

Since we are splitting the data table into two at each step without deleting any column, in the worst-case scenario, there can be a total of $(N-1)$ nodes in the decision tree. Hence, in the worst-case scenario, the code will run a total of $N-1$ times. In every iteration of the code, each of the columns present in the table has to be passed to the information gain function to obtain the column with the best information gain in order to carry out the appropriate spilt. Thus, this will occur a total of $M$ times. The time complexity of the information function will be $O(N^2)$ since it takes $O(N)$ to calculate the information for a given split, and there are a total of $(N-1)$ splits. Thus, each iteration will have a time complexity of $O(N^2M)$. 

Hence the complexity for the case of real inputs will be $O(N^3M)$. This is the Learning Time Complexity.

The Prediction Time Complexity depends on the depth of the tree. In the worst case, the sample input to be predicted would reach the leaf at the lowest level - the
depth of the tree. This prediction would have to go through all the $(N-1)$ splits and so the Prediction Time Complexity would be $O(N)$.

<h4>2. Discrete input values: </h4>

The total number of times the code is executed in the program depends on the relative sizes of N and M. The total number of unique input samples that can be made from the attributes depends on the number of unique attribute values in each column. It will be equal to the product of all $(d_i)’s$ for all values of i. Thus, the number of leaves in the decision tree can not exceed this value. On the other hand, for a given number of output leaves N, the number of non-leaf nodes (decision nodes) in the tree can’t exceed $(N-1)$, since every node should either be a child node or spilt at least two times. Since the code runs as many times as the number of non-leaf nodes in the decision tree in the worst-case scenario, it will run a total of $min(D, N-1)$ times, where $D$ is the product of all $d_i$’s. 

In the case that the product of all unique attributes from each of the columns is lesser than the sample size $(N)$: The total number of all possible combinations for a row in the input set is equal to the product of all the unique attributes from each of the columns. Thus, in the case that this value is less than the total number of samples, the sets of samples taking the same value(s) for all attributes will be clubbed together into a single leaf and hence the maximum number of leaves will be the maximum number of unique samples which is equal to the product of all $(d_i)’s$. So, the maximum possible iteration will be one less than the number of leaves which is the product value, even in the worst-case scenario.

In the case that the product of all unique attributes from each of the columns is greater than the sample size $(N)$: Each sample could be unique in the worst case and we could have a perfect classification with $N$ leaves. So the internal nodes would be $(N-1)$ and we would have the maximum possible iterations as $(N-1)$, so this is $O(N)$

In each of the iterations, the code will pass all the columns present in the table to the information gain function. Thus, the information gain function is run a total of $O(M)$ times in each of the iterations. The complexity of this function is equal to the size of the input series since it has to just pass the series twice in order to find the information gain. Thus, each iteration of the code will have a time complexity of $O(NM)$. 

Hence the learning time complexity for the case of discrete inputs will be $O(min(D, N-1)(NM))$.

The prediction time complexity for this case would depend on the depth of the tree. In the worst case, every one of the $M$ attributes would occur as a decision node in the path from root to leaf and so we would ask $M$ questions with the depth of the tree being equal to $M$. So, the prediction time complexity would be $O(M)$. 


<h3> Plots </h3>

<ol>
  <li><h4> Real Input, Real Output </h4></li>
    <ol>
      <li> Learning Average Time </li>
      <li> Learning Standard Deviation </li>
      <li> Predicting Average Time </li>
      <li> Predicting Standard Deviation </li>
    </ol>
      
  
  <li><h4> Real Input, Discrete Output </h4></li>
    <ol>
      <li> Learning Average Time </li>
      <li> Learning Standard Deviation </li>
      <li> Predicting Average Time </li>
      <li> Predicting Standard Deviation </li>
    </ol>
      
  
  
  <li><h4> Discrete Input, Real Output </h4></li>
    <ol>
      <li> Learning Average Time </li>
      <li> Learning Standard Deviation </li>
      <li> Predicting Average Time </li>
      <li> Predicting Standard Deviation </li>
    </ol>
      
  
  <li><h4> Discrete Input, Discrete Output </h4></li>
    <ol>
      <li> Learning Average Time </li>
      <li> Learning Standard Deviation </li>
      <li> Predicting Average Time </li>
      <li> Predicting Standard Deviation </li>
    </ol>     
</ol>
  

