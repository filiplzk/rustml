use csv::ReaderBuilder;

use rand::seq::SliceRandom;
use rustml::*;

fn from_csv(path: &str) -> Vec<(Matrix, usize)> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_path(path).unwrap();

    let mut all_rows: Vec<Vec<f32>> = Vec::new();

    for result in rdr.records() {
        let string_record = result.unwrap();

        let f_vec_result: Result<Vec<f32>, _> = string_record
            .iter()
            .map(|field| field.trim().parse::<f32>())
            .collect();

        match f_vec_result {
            Ok(f_vec) => all_rows.push(f_vec),
            Err(_) => ()
        }
    }

    let mut dataset: Vec<(Matrix, usize)> = Vec::new();

    for row in &all_rows {
        let mut pixels: Vec<f32> = row.clone();
        let label = pixels.remove(0) as usize;
        pixels = pixels.iter().map(|x| x / 256.0).collect();
        dataset.push((Matrix::new(&[&pixels]), label));
    }

    dataset
} 


fn main() {
    let mut rng: rand::prelude::ThreadRng = rand::thread_rng();
    
    // configuration --------

    // datasets
    let num_train_samples = 60000;
    let num_test_samples = 10000;
    let train_shuffle = true;
    let test_shuffle = true;
    
    // network
    let mut net = nn::Sequential::new();
    net.add(nn::Linear::new(&mut rng, 28*28, 32));
    net.add(nn::ReLU::new());
    net.add(nn::Linear::new(&mut rng, 32, 32));
    net.add(nn::ReLU::new());
    net.add(nn::Linear::new(&mut rng, 32, 32));
    net.add(nn::ReLU::new());
    net.add(nn::Linear::new(&mut rng, 32, 10));
    
    // optimizer
    let mut optimizer = optim::SGD::new(net.params(), 1e-3, 0.9);

    // logging
    let group_size = 100;

    // ----------------------

    
    // data
    let mut train_dataset = from_csv("examples/data/MNIST_CSV/mnist_train.csv");
    let mut test_dataset = from_csv("examples/data/MNIST_CSV/mnist_test.csv");
    
    if train_shuffle {
        train_dataset.shuffle(&mut rng);
    }
    if test_shuffle {
        test_dataset.shuffle(&mut rng);
    }
    
    let num_train_samples = num_train_samples.min(train_dataset.len());
    let num_test_samples = num_test_samples.min(test_dataset.len());

    train_dataset = train_dataset[0..num_train_samples].to_vec();
    test_dataset = test_dataset[0..num_test_samples].to_vec();

    println!("Datasets loaded");

    // training
    let mut correct = 0;
    let mut idx = 0;
    for (data, label) in &train_dataset {
        let out = net.forward(data);
        let (loss, probs) = functional::cross_entropy_loss(&out, *label);

        loss.backward();
        optimizer.step();
        loss.reset_graph();

        let (pred, confidence) = (0..10)
            .map(|i: usize| probs[(0, i)].data().val)
            .enumerate()
            .max_by(|&(_, a), &(_, b)| a.partial_cmp(&b).unwrap())
            .unwrap();

        if pred == *label {
            correct += 1;
        }
        
        if (idx + 1) % group_size == 0 {
            println!("i={}, loss={:.3}, label={label}, pred={pred}, accuracy={:.3}%, confidence={:.3}%",
                idx+1, loss.data().val, correct as f32 / group_size as f32 * 100.0, confidence * 100.0);
            correct = 0;
        }

        idx += 1;
    }

    // testing
    for (data, label) in &test_dataset {
        let out = net.forward(data);
        let (_, probs) = functional::cross_entropy_loss(&out, *label);

        let (pred, _) = (0..10)
            .map(|i: usize| probs[(0, i)].data().val)
            .enumerate()
            .max_by(|&(_, a), &(_, b)| a.partial_cmp(&b).unwrap())
            .unwrap();

        if pred == *label {
            correct += 1;
        }
    }

    let test_acccuracy = correct as f32 / num_test_samples as f32;
    println!("Test accuracy: {:.2}%", test_acccuracy * 100.0);
}
