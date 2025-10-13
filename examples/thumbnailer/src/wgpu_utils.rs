use wgpu::{
    Device, DeviceDescriptor, Extent3d, Features, Instance, InstanceDescriptor, Limits,
    MemoryHints, PowerPreference, Queue, RequestAdapterOptions, Texture, TextureDescriptor,
    TextureFormat, TextureUsages, TextureView, Trace,
};

pub fn create_wgpu_device() -> Result<(Device, Queue), Box<dyn std::error::Error>> {
    let instance = Instance::new(&InstanceDescriptor::default());

    let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
        power_preference: PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        ..Default::default()
    }))
    .expect("Failed to find an appropriate adapter");

    let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor {
        label: None,
        // Request the feature for wireframe rendering
        required_features: Features::POLYGON_MODE_LINE, // https://docs.rs/wgpu/latest/wgpu/struct.FeaturesWGPU.html#associatedconstant.POLYGON_MODE_LINE
        required_limits: Limits::default(),
        memory_hints: MemoryHints::Performance,
        trace: Trace::Off,
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
    }))
    .expect("Failed to create device");

    Ok((device, queue))
}

pub fn create_render_texture(
    device: &Device,
    width: u32,
    height: u32,
    transparent: bool,
) -> (Texture, TextureView) {
    let texture_size = Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };

    let texture = device.create_texture(&TextureDescriptor {
        label: Some("Thumbnail Render Texture"),
        size: texture_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: if transparent {
            TextureFormat::Rgba8UnormSrgb
        } else {
            TextureFormat::Bgra8UnormSrgb
        },
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    (texture, texture_view)
}
