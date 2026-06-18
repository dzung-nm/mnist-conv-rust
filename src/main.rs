use mnist_conv_rust::load_mnist;
use mnist_conv_rust::Layer;
use mnist_conv_rust::conv_pool_layer::*;
use mnist_conv_rust::network::*;
use mnist_conv_rust::fully_connected_layer::FullyConnectedLayer;
use mnist_conv_rust::softmax_layer::SoftmaxLayer;
use mnist_conv_rust::types::Dataset;

fn main() {
    let mnist_data = load_mnist().expect("Failed to load MNIST dataset");

    let size = 50000; // Test with a smaller subset of the data for faster training
    let data = Dataset {
        training: mnist_data.training.into_iter().take(size).collect(),
        test: mnist_data.test,
        validation: mnist_data.validation,
    };

    println!(
        "Training data size: {} samples, {} validation samples, {} test samples",
        data.training.len(),
        data.test.len(),
        data.validation.len()
    );

    let net_options = NetOptions {
        max_epochs: 10,
        mini_batch_size: 20,
        eta: 0.1,
        regularization_l2: 5.0,
        ..NetOptions::default()
    };
    
    let fused_layer_config = ConvPoolLayerConfig {
        input: (1, 28, 28),
        kernel_size: (5, 5),
        num_filters: 6,
        stride: 1,
        padding: 0,
        pool_size: (2, 2),
        pool_stride: 2,
    };
    
    let layers: Vec<Box<dyn Layer>> = vec![
        Box::new(ConvPoolLayer::new(&fused_layer_config)), // → 6×12×12 = 864
        Box::new(FullyConnectedLayer::new_with_dropout(864, 30, 0.5)), // 50% dropout
        Box::new(SoftmaxLayer::new(30, 10)),
    ];

    let mut network = Network::new(layers, net_options);

    println!("===============================");
    network.show_me();
    println!("===============================");
    println!("Training...");

    network.sdg(&data);
}
