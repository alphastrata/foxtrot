use clap::Parser;
use std::time::SystemTime;
use step::step_file::StepFile;

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Input STEP file to parse
    input: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let input = &args.input;

    let start = SystemTime::now();

    let data = std::fs::read(input)?;
    let flat = StepFile::strip_flatten(&data);
    let entities = StepFile::parse(&flat);
    println!("Got {} entities", entities.0.len());

    let end = SystemTime::now();
    let since_the_epoch = end.duration_since(start).expect("Time went backwards");
    println!("time {:?}", since_the_epoch);
    Ok(())
}
