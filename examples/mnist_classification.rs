use std::os::unix::process;

use rustml::*;
use rand::seq::SliceRandom;
use csv::ReaderBuilder;


fn from_csv(path: &str, batch_size: usize) -> Vec<(Tensor<f32>, Tensor<usize>)> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
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

    let mut dataset: Vec<(Tensor<f32>, Tensor<usize>)> = Vec::new();
    let mut batch_flat: Vec<f32> = Vec::new();
    let mut batch_labels: Vec<usize> = Vec::new();

    for (idx, row) in all_rows.iter().enumerate() {
        let mut pixels: Vec<f32> = row.clone();
        let label = pixels.remove(0) as usize;
        pixels = pixels.iter().map(|x| x / 256.0).collect();
        batch_flat.extend_from_slice(&pixels);
        batch_labels.push(label);
        if (idx + 1) % batch_size == 0 || idx + 1 == all_rows.len() {
            let b = (idx % batch_size) + 1;
            dataset.push((Tensor::from_flat([b, 1, pixels.len()], &batch_flat), Tensor::from_flat([b], &batch_labels)));
            batch_flat.clear();
            batch_labels.clear();
        } 
    }
    dataset
} 


fn main() {
    let mut rng = rand::rng();
    
    // configuration --------

    // datasets
    let num_train_samples = 60000;
    let num_test_samples = 100;
    let train_batch_size = 2;
    let train_shuffle = true;
    let test_shuffle = true;
    let epochs = 2;
    
    // network
    let mut net = nn::Sequential::new();
    net.add(nn::Linear::new_he(&mut rng, 28*28, 32));
    net.add(nn::ReLU::new());
    net.add(nn::Linear::new_he(&mut rng, 32, 32));
    net.add(nn::ReLU::new());
    net.add(nn::Linear::new_he(&mut rng, 32, 32));
    net.add(nn::ReLU::new());
    net.add(nn::Linear::new_he(&mut rng, 32, 10));
    net.add(nn::Softmax::new());

    // optimizer
    let mut optimizer = optim::SGD::new(net.params(), 2e-4, 0.9);

    // logging
    let group_size = 100;

    // ----------------------

    
    // data
    let mut train_dataset = from_csv("examples/data/MNIST_CSV/mnist_train.csv", train_batch_size);
    let mut test_dataset = from_csv("examples/data/MNIST_CSV/mnist_test.csv", 1);
    
    if train_shuffle {
        train_dataset.shuffle(&mut rng);
    }
    if test_shuffle {
        test_dataset.shuffle(&mut rng);
    }

    train_dataset = train_dataset[0..num_train_samples / train_batch_size].to_vec();
    test_dataset = test_dataset[0..num_test_samples].to_vec();

    println!("Datasets loaded");

    // training
    let mut loss_sum = 0.0;
    for epoch in 1..=epochs {
        for (idx, (data, labels)) in train_dataset.iter().enumerate() {            
            let tgt = labels.stack_new_dim(1, 1).one_hot(10).cast::<f32>();     
            let probs = net.forward(data);

            let batch_losses = functional::cross_entropy_loss(&probs, &tgt);
            let loss = batch_losses.sum_all() / Tensor::fill([1], batch_losses.size() as f32);          

            loss.backward();
            optimizer.step();
            loss.zero_grad();
            
            loss_sum += loss.item();
            if (idx + 1) % group_size == 0 {
                println!("Epoch {}/{}, ({}-{})/{}: average loss={}", epoch, epochs, idx-group_size+2, idx+1, train_dataset.len(), loss_sum / group_size as f32);
                loss_sum = 0.0;
            }
        }
    }
        
    // testing
    // let mut correct = 0;
    // for (data, label) in &test_dataset {
    //     let class = label.flat()[0].val();

    //     let tgt = Tensor::<f32>::zeros([1, 1, 10]);
    //     for (idx, scalar) in label.flat().iter().enumerate() {
    //         tgt[[idx, 0, scalar.val()]].set(1.0);
    //     }

    //     let logits = net.forward(data);
    //     let probs = functional::softmax(&logits);

    //     let mut cur_max = 0;
    //     for (idx, logit) in probs.flat().iter().enumerate() {
    //         if logit.val() > probs.flat()[cur_max].val() {
    //             cur_max = idx;
    //         }
    //     }

    //     if cur_max == class {
    //         correct += 1;
    //     }
    // }

    // println!("Accuracy: {:.2}% ({}/{})", correct as f32 / num_test_samples as f32 * 100.0, correct as f32, num_test_samples as f32);
}
