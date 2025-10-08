use clap::Parser;

use step::step_file::StepFile;
use triangulate::triangulate::triangulate;

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// STL file to output
    #[clap(short, long = "out")]
    output: String,

    /// Input STEP file to convert
    input: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args = Args::parse();
    let input = &args.input;

    let start = std::time::SystemTime::now();
    let data = std::fs::read(input)?;
    let flat = StepFile::strip_flatten(&data);
    let entities = StepFile::parse(&flat);
    let end = std::time::SystemTime::now();
    let since_the_epoch = end.duration_since(start).expect("Time went backwards");
    println!("Loaded + parsed in {:?}", since_the_epoch);

    let start = std::time::SystemTime::now();
    let tri = triangulate(&entities);
    let end = std::time::SystemTime::now();
    let since_the_epoch = end.duration_since(start).expect("Time went backwards");
    println!("Triangulated in {:?}", since_the_epoch);

    tri.0.save_stl(&args.output)?;

    Ok(())
}
