use plotters::prelude::*;

/// Plots a vector of (x, y) points and saves it as a PNG
pub fn plot_data(filename: &str, title: &str, data: &[(f32, f32)]) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let (x_min, x_max) = data.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &(x, _)| {
        (min.min(x), max.max(x))
    });
    let (_, y_max) = data.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &(_, y)| {
        (min.min(y), max.max(y))
    });

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_min..x_max, 0.0..y_max)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(data.to_vec(), &RED))?;

    Ok(())
}