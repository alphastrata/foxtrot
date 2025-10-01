use clap::Parser;
use step::step_file::StepFile;

pub fn to_dot(s: &StepFile) -> String {
    let mut out = "digraph {\n".to_owned();
    for (i, e) in s.0.iter().enumerate() {
        let d = format!("{:?}", e);
        let name = d.split("(").next().unwrap();

        out += &format!("  e{} [ label = \"#{}: {}\" ];\n", i, i, name);
        for j in e.upstream() {
            out += &format!("  e{} -> e{};\n", i, j);
        }
    }
    out += "}";
    out
}

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Output dot file (optional, prints to stdout if not provided)
    #[clap(short, long = "out")]
    output: Option<String>,
    
    /// Input STEP file to convert
    input: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let input = &args.input;

    let start = std::time::SystemTime::now();
    let data = std::fs::read(input)?;
    let flat = StepFile::strip_flatten(&data);
    let entities = StepFile::parse(&flat);
    let end = std::time::SystemTime::now();
    let since_the_epoch = end.duration_since(start).expect("Time went backwards");
    println!("Loaded + parsed in {:?}", since_the_epoch);

    let dot = to_dot(&entities);
    if let Some(output) = &args.output {
        std::fs::write(output, dot)?;
    } else {
        println!("{}", dot);
    }
    Ok(())
}
