use ndarray::Array2;

/// Represents a training data item with image and one-hot encoded label
pub struct TrainingItem(pub Array2<f64>, pub Array2<f64>); // e.g. (image (784,1), one-hot label (10,1))

/// Represents a test/validation data item with image and raw label
pub struct TestItem(pub Array2<f64>, pub u8); // e.g. (image (784,1), label)

#[derive(PartialEq, Debug)]
pub enum DatasetType {
    Mnist,
    Cifar10,
}

pub struct Dataset {
    pub training: Vec<TrainingItem>,
    pub validation: Vec<TestItem>,
    pub test: Vec<TestItem>,
    
    // Indicates whether the dataset is MNIST or CIFAR-10
    pub dataset_type: DatasetType,
    
    // A vector of label names corresponding to the indices of the output layer.
    // Example, for cifar 10: ["airplane", "automobile", "bird", "cat",...] 
    pub labels: Vec<String>,

    // Optional function to create augmented training data
    pub new_augmented_data: Option<fn(Vec<&TrainingItem>, multiplier: usize) -> Vec<TrainingItem>>,
}
