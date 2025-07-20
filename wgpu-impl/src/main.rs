

use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};
use winit::event::WindowEvent;
use wgpu::util::DeviceExt;

mod gpucontext;
use gpucontext::GpuContext;

struct State<'a> {
    gpu: GpuContext<'a>,
}

impl<'a> State<'a> {
    async fn new(window: Arc<Window>) -> Self {
        let gpu = GpuContext::new(window).await;
        Self { gpu }
    }
    fn run_compute(&self) {
        let device = &self.gpu.device;
        let queue = &self.gpu.queue;
        let input_data = vec![1u32, 2, 3, 4];
        let buffer_size = (input_data.len() * std::mem::size_of::<u32>()) as wgpu::BufferAddress;

        let storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Storage Buffer"),
            contents: bytemuck::cast_slice(&input_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Load SPIR-V shader using ShaderSource::SpirV
        let shader = device.create_shader_module(wgpu::include_spirv!("./shader.spv"));

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            cache: None,
            compilation_options: Default::default(),
        });
        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: storage_buffer.as_entire_binding(),
            }],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(input_data.len() as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&storage_buffer, 0, &output_buffer, 0, buffer_size);
        queue.submit(Some(encoder.finish()));
        let buffer_slice = output_buffer.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        device.poll(wgpu::PollType::Wait);
        futures_lite::future::block_on(async {
            rx.receive().await;
        });
        let data = buffer_slice.get_mapped_range();
        let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        output_buffer.unmap();
        println!("Compute shader output: {:?}", result);
    }
}

struct App<'a> {
    window: Option<Arc<Window>>,
    state: Option<State<'a>>,
}

impl<'a> Default for App<'a> {
    fn default() -> Self {
        Self {
            window: None,
            state: None,
        }
    }
}

impl<'a> ApplicationHandler for App<'a> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(Window::default_attributes().with_visible(false))
            .unwrap();
        self.window = Some(Arc::new(window));
        self.state = Some(futures_lite::future::block_on(State::new(self.window.as_ref().unwrap().clone())));
        // Run compute shader and exit
        self.state.as_ref().unwrap().run_compute();
        event_loop.exit();
    }
    fn window_event(&mut self, _event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        if let WindowEvent::CloseRequested = event {
            _event_loop.exit();
        }
    }
}

fn main() {
    let mut app = App::default();
    let event_loop = EventLoop::new().unwrap();
    event_loop.run_app(&mut app).unwrap();
}
