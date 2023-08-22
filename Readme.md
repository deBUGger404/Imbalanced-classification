# Imbalanced classification

Welcome to the Imbalanced data classification repository! This repository contains code related to the classification of products into different categories using deep learning techniques. The project is divided into three main parts: dataset creation, model creation for different classification tasks, and utilizing transfer learning for better training efficiency.

## Codes Info

1. **dataset**: This directory contains the code used to create the dataset. It's important to review this code before running the modeling code to understand how the dataset was generated.

2. **model_part1**: In this section, a deep learning model has been developed to classify products as either packed or loose.

3. **model_part2**: This section is dedicated to the solution for part B. Here, a deep learning model has been constructed to classify products into coarse categories within the packed product group.

4. **model_part3**: For part C, an imbalanced class problem is addressed. Data augmentation techniques and a deep learning model have been employed to effectively classify imbalanced classes within the loose product category.

## Additional Information

- One-hot Encoding: The labels have been one-hot encoded and saved in the file `ohe_final.pkl`.

- Saved Models: The trained model objects for the respective parts are saved as `part1.h5`, `part2.h5`, and `part3.h5`.

- Reproducibility: To ensure reproducibility, seed fixing methods have been implemented at the beginning of the code.

- TensorFlow and tf.data: The TensorFlow library was used for creating and training the deep learning models. The `tf.data` module was utilized to implement data prefetching, which significantly reduces training time.

- Transfer Learning: Transfer learning techniques were employed to leverage pre-trained models, enhancing model performance and reducing training time.

Feel free to explore the individual directories for detailed code implementations and further explanations. If you have any questions or suggestions, please feel free to open an issue or contact us.

Happy coding! :thumbsup:
