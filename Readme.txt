codes info:

1. dataset: dataset creation code please have look once before run modeling code
2. model_part1: For part A solution, created deep learning model to classify between packed vs loose product
3. model_part2: For part B solution, created deep learning model to classify coarse classes of packed product
4. model_part3: For part C solution, created and leveraged data augmentationa and deep learning model to classify imbalanced classes of loose product

// there is a one hot encoded object which i used to encode labels: ohe_final.pkl
// And I saved model object for respective model_parts: part1.h5, part2.h5 and part3.h5
// used seed fixing method to reproducing same results again on the starting of the code
// used tensorflow`s tf.data to leverage the prefecthing method to decrease the training time
// used transfer learning to train model better and in less time


