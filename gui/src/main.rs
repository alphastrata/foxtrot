use std::time::SystemTime;
use winit::{
    event_loop::EventLoop,
    window::Window,
};

pub(crate) mod app;
pub(crate) mod backdrop;
pub(crate) mod camera;
pub(crate) mod model;

use crate::app::App;
use clap::Parser;
use triangulate::mesh::Mesh;

#[derive(clap::Parser)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Input STEP file to render
    #[clap(value_parser)]
    input: String,
}

struct State {
    app: Option<App>,
    window: Option<Window>,
    queue: Option<wgpu::Queue>,
    loader: Option<std::thread::JoinHandle<Mesh>>,
    surface: Option<wgpu::Surface<'static>>,
    device: Option<wgpu::Device>,
    adapter: Option<wgpu::Adapter>,
}

impl State {
    fn new(window: Window, loader: std::thread::JoinHandle<Mesh>) -> Self {
        Self {
            app: None,
            window: Some(window),
            queue: None,
            loader: Some(loader),
            surface: None,
            device: None,
            adapter: None,
        }
    }
}

fn run(
    start: SystemTime,
    event_loop: EventLoop<()>,
    loader: std::thread::JoinHandle<Mesh>,
) {
    use winit::application::ApplicationHandler;
    use winit::event::WindowEvent;
    use winit::event_loop::ActiveEventLoop;
    
    struct AppHandler {
        app: Option<App>,
        window: Option<Window>,
        queue: Option<wgpu::Queue>,
        loader: Option<std::thread::JoinHandle<Mesh>>,
        surface: Option<wgpu::Surface<'static>>, // This is still problematic but using unsafe transmute as workaround
        device: Option<wgpu::Device>,
        adapter: Option<wgpu::Adapter>,
        start_time: SystemTime,
    }

    impl ApplicationHandler<()> for AppHandler {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            // Create window on resumed, which is the winit 0.30 pattern
            if self.window.is_none() {
                let window = event_loop.create_window(
                    winit::window::WindowAttributes::default()
                        .with_title("Foxtrot")
                ).unwrap();
                self.window = Some(window);
            }
            
            // Request redraw after window is created
            if let Some(ref window) = self.window {
                window.request_redraw();
            }
        }

        fn window_event(
            &mut self,
            event_loop: &ActiveEventLoop,
            _window_id: winit::window::WindowId,
            event: WindowEvent,
        ) {
            if let WindowEvent::CloseRequested = event {
                event_loop.exit();
                return;
            }

            // Initialize graphics resources if not done yet and we have a window
            if self.app.is_none()
                && let Some(window) = &self.window {
                    // Create graphics resources
                    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                        backends: wgpu::Backends::all(),
                        flags: wgpu::InstanceFlags::empty(),
                        ..Default::default()
                    });
                    
                    // Create surface
                    let surface = 
                        instance.create_surface(window).expect("Could not create surface");
                    
                    let adapter = pollster::block_on(instance
                        .request_adapter(&wgpu::RequestAdapterOptions {
                            power_preference: wgpu::PowerPreference::HighPerformance,
                            force_fallback_adapter: false,
                            ..Default::default()
                        }))
                        .expect("Failed to find an appropriate adapter");

                    let (device, queue) = pollster::block_on(adapter
                        .request_device(
                            &wgpu::DeviceDescriptor {
                                label: None,
                                required_features: wgpu::Features::empty(),
                                required_limits: wgpu::Limits::default(),
                                memory_hints: wgpu::MemoryHints::Performance,
                                trace: wgpu::Trace::Off,
                                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                            },
                        ))
                        .expect("Failed to create device");

                    let size = window.inner_size();
                    
                    // First create the app with the surface (to get capabilities)
                    let app = App::new(
                        self.start_time, 
                        size, 
                        adapter.clone(), 
                        &surface, // Pass reference to get capabilities
                        device.clone(), 
                        self.loader.take().unwrap()
                    );
                    
                    // Configure the surface
                    let surface_config = wgpu::SurfaceConfiguration {
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                        format: app.surface_format,
                        width: size.width,
                        height: size.height,
                        present_mode: wgpu::PresentMode::Fifo,
                        alpha_mode: wgpu::CompositeAlphaMode::Auto,
                        view_formats: vec![],
                        desired_maximum_frame_latency: 2,
                    };
                    surface.configure(&device, &surface_config);
                    
                    self.app = Some(app);
                    self.queue = Some(queue);
                    self.device = Some(device);
                    self.adapter = Some(adapter);
                    // Workaround lifetime issue - extend surface lifetime unsafely
                    self.surface = Some(unsafe { std::mem::transmute(surface) });
                }

            // Process window event with app
            if let Some(ref mut app) = self.app {
                use app::Reply;
                match app.window_event(event) {
                    Reply::Continue => (),
                    Reply::Quit => event_loop.exit(),
                    Reply::Redraw => {
                        if let (Some(surface), Some(queue)) = (&self.surface, &self.queue)
                            && app.redraw(surface, queue)
                                && let Some(window) = &self.window {
                                    window.request_redraw();
                                }
                    }
                }
            }
        }

        fn device_event(
            &mut self,
            _event_loop: &ActiveEventLoop,
            _device_id: winit::event::DeviceId,
            event: winit::event::DeviceEvent,
        ) {
            if let Some(ref mut app) = self.app {
                app.device_event(event);
            }
        }

        fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
            // Optional: Called repeatedly when the event loop is idle
        }
    }

    let mut app_handler = AppHandler {
        app: None,
        window: None,
        queue: None,
        loader: Some(loader),
        surface: None,
        device: None,
        adapter: None,
        start_time: start,
    };

    event_loop.run_app(&mut app_handler).unwrap();
}

fn main() {
    env_logger::init();

    let args = Args::parse();
    let input = args.input;

    // Kick off the loader thread immediately, so that the STEP file is parsed
    // and triangulated in the background while we wait for a GPU context
    let loader = std::thread::spawn(|| {
        println!("Loading mesh!");
        use step::step_file::StepFile;
        use triangulate::triangulate::triangulate;

        let data = std::fs::read(input).expect("Could not open file");
        let flat = StepFile::strip_flatten(&data);
        let step = StepFile::parse(&flat);
        let (mesh, _stats) = triangulate(&step);
        mesh
    });

    let event_loop = EventLoop::new().unwrap();
    run(SystemTime::now(), event_loop, loader);
}
