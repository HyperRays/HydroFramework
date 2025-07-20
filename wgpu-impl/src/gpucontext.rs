use std::sync::Arc;

use wgpu::{
    Adapter, Device, Instance, InstanceDescriptor, Queue, Surface, SurfaceConfiguration,
    TextureFormat,
};
use winit::{dpi::PhysicalSize, window::Window};

/// ---------------------------------------------------------------------------
/// 1. GpuContext - Manages adapter, device, queue, surface, & config
/// ---------------------------------------------------------------------------

#[derive(Debug)]
#[allow(unused)]
pub struct GpuContext<'a> {
    pub instance: Instance,
    pub surface: Surface<'a>,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    pub config: SurfaceConfiguration,
    pub swapchain_format: TextureFormat,
}

impl<'a> GpuContext<'a> {
    /// Create a new instance + surface + adapter + device + queue + config
    pub async fn new(window: Arc<Window>) -> Self {
        // Decide whether to enable debug flags
        let instance_descriptor = InstanceDescriptor::default();

        // Create the WGPU instance
        let instance = wgpu::Instance::new(&instance_descriptor);

        // Create the surface from the window
        let surface = instance.create_surface(window.clone()).unwrap();

        // Pick an adapter that works with the surface
        let adapter = instance
            .enumerate_adapters(wgpu::Backends::all())
            .into_iter()
            .find(|a| a.is_surface_supported(&surface))
            .expect("Failed to find an adapter that supports the surface");

        // Request device and queue
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::PUSH_CONSTANTS
                    | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                    | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM
                    | wgpu::Features::CLEAR_TEXTURE
                    | wgpu::Features::TIMESTAMP_QUERY
                    | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
                required_limits: wgpu::Limits {
                    max_bind_groups: 6,
                    max_push_constant_size: 4 * 5,
                    ..wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits())
                },
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("Failed to create device");

        // Create a surface configuration
        let size = window.inner_size();
        let mut config = surface
            .get_default_config(&adapter, size.width, size.height)
            .expect("Failed to get default config");

        config.format = wgpu::TextureFormat::Bgra8UnormSrgb;
        // For example:
        // config.present_mode = wgpu::PresentMode::Immediate;

        // Swapchain format is typically the first supported format
        let swapchain_format = config.format;

        // Configure surface
        surface.configure(&device, &config);

        Self {
            instance,
            surface,
            adapter,
            device,
            queue,
            config,
            swapchain_format,
        }
    }

    /// Update the surface configuration on resize
    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }
}
