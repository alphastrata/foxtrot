use clap::Parser;
use rusttype::{Font, OutlineBuilder, Scale, point};

const BEZIER_RESOLUTION: usize = 4;

#[derive(Default)]
struct Builder {
    points: Vec<(f64, f64)>,
    contours: Vec<Vec<usize>>,
    x: f32,
    y: f32,
    dx: f32,
    dy: f32,
}

impl Builder {
    fn set_offset(&mut self, dx: f32, dy: f32) {
        self.dx = dx;
        self.dy = dy;
    }

    fn set_position(&mut self, x: f32, y: f32) {
        self.x = x;
        self.y = -y;
        self.points
            .push(((x + self.dx) as f64, (self.dy - y) as f64));
    }
}

impl OutlineBuilder for Builder {
    fn move_to(&mut self, x: f32, y: f32) {
        // Begin a new contour
        self.contours.push(vec![self.points.len()]);
        self.set_position(x, y);
    }

    fn line_to(&mut self, x: f32, y: f32) {
        // Push a move to this point into the last contour
        self.contours.last_mut().unwrap().push(self.points.len());
        self.set_position(x, y);
    }

    fn close(&mut self) {
        // Remove last coordinate + point (which is a duplicate), then reassign
        let c = self.contours.last_mut().unwrap();
        *c.last_mut().unwrap() = c[0];
        self.points.pop().unwrap();

        // Leave position unchanged since we're going to start a new contour
        // shortly (if all is behaving well)
    }

    fn quad_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32) {
        let x0 = self.x;
        let y0 = -self.y;
        for i in 1..=BEZIER_RESOLUTION {
            let t = i as f32 / (BEZIER_RESOLUTION as f32);
            let f = |a, b, c| (1.0 - t).powf(2.0) * a + 2.0 * (1.0 - t) * t * b + t.powf(2.0) * c;
            self.line_to(f(x0, x1, x2), f(y0, y1, y2));
        }
    }

    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x3: f32, y3: f32) {
        let x0 = self.x;
        let y0 = -self.y;

        for i in 1..=BEZIER_RESOLUTION {
            let t = i as f32 / (BEZIER_RESOLUTION as f32);
            let f = |a, b, c, d| {
                (1.0 - t).powf(3.0) * a
                    + 3.0 * (1.0 - t).powf(2.0) * t * b
                    + 3.0 * (1.0 - t) * t.powf(2.0) * c
                    + t.powf(3.0) * d
            };
            self.line_to(f(x0, x1, x2, x3), f(y0, y1, y2, y3));
        }
    }
}

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Path to the font TTF
    #[clap(short, long = "font", default_value = "/Library/Fonts/Georgia.ttf")]
    font: String,
    
    /// SVG file to output
    #[clap(short, long = "out")]
    output: Option<String>,
    
    /// Check invariants after each step (slow)
    #[clap(short, long)]
    check: bool,
    
    /// Text to triangulate
    #[clap(short, long, default_value = "hello, world")]
    text: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let font = {
        let data = std::fs::read(&args.font)?;
        Font::try_from_vec(data).unwrap()
    };

    // Load the font glyphs into the triangulation builder
    let mut builder = Builder::default();
    let scale = Scale::uniform(10.0);
    for g in font.layout(&args.text, scale, point(0.0, 0.0)) {
        let pos = g.position();
        builder.set_offset(pos.x, pos.y);
        g.unpositioned().build_outline(&mut builder);
    }

    // Then, do the work of triangulation
    let now = std::time::Instant::now();
    let mut t = cdt::Triangulation::new_from_contours(&builder.points, &builder.contours)?;
    while !t.done() {
        t.step()?;
        if args.check {
            t.check();
        }
    }
    let result = t.triangles().collect::<Vec<_>>();
    let elapsed = now.elapsed();

    eprintln!(
        "    Triangulated '{}' in {}.{}s.\n    Generated {} triangles.",
        args.text,
        elapsed.as_secs(),
        elapsed.subsec_millis(),
        result.len(),
    );

    if let Some(output) = &args.output {
        eprintln!("    Saving {}", output);
        t.save_debug_svg(output).expect("Could not save SVG");
    }

    Ok(())
}
