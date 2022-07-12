#![feature(try_blocks)]
use ash::{
    vk::{
        Result as VkResult,
        FALSE,
        TRUE,
        make_api_version,
        StructureType,

        SurfaceKHR,

        ApplicationInfo,
        InstanceCreateInfo,
        InstanceCreateFlags,

        DebugUtilsMessengerEXT,
        DebugUtilsMessengerCreateInfoEXT,
        DebugUtilsMessengerCreateFlagsEXT,
        DebugUtilsMessageSeverityFlagsEXT,
        DebugUtilsMessageTypeFlagsEXT,
        DebugUtilsMessengerCallbackDataEXT,

        PhysicalDevice,
        QueueFlags,
        DeviceQueueCreateInfo,
        DeviceQueueCreateFlags,
        DeviceCreateInfo,
        DeviceCreateFlags,

        Format,
        SurfaceFormatKHR,
        ColorSpaceKHR,
        PresentModeKHR,
        SurfaceCapabilitiesKHR,
        Extent2D,
        SwapchainCreateInfoKHR,
        SwapchainCreateFlagsKHR,
        ImageUsageFlags,
        SharingMode,
        CompositeAlphaFlagsKHR,
        SwapchainKHR,

        AttachmentDescription,
        AttachmentDescriptionFlags,
        SampleCountFlags,
        AttachmentLoadOp,
        AttachmentStoreOp,
        ImageLayout,
        AttachmentReference,
        SubpassDescription,
        AccessFlags,
        DependencyFlags,
        SubpassDescriptionFlags,
        PipelineBindPoint,
        RenderPassCreateInfo,
        RenderPassCreateFlags,
        RenderPass,

        ShaderModule,
        ShaderModuleCreateInfo,
        ShaderModuleCreateFlags,

        PipelineShaderStageCreateInfo,
        PipelineShaderStageCreateFlags,
        ShaderStageFlags,
        PipelineVertexInputStateCreateInfo,
        PipelineVertexInputStateCreateFlags,
        PipelineInputAssemblyStateCreateInfo,
        PipelineInputAssemblyStateCreateFlags,
        PrimitiveTopology,
        PipelineRasterizationStateCreateInfo,
        PipelineRasterizationStateCreateFlags,
        CullModeFlags,
        FrontFace,
        PipelineMultisampleStateCreateInfo,
        PipelineMultisampleStateCreateFlags,
        PipelineColorBlendAttachmentState,
        ColorComponentFlags,
        BlendFactor,
        BlendOp,
        LogicOp,
        DynamicState,
        PolygonMode,
        PipelineColorBlendStateCreateInfo,
        PipelineColorBlendStateCreateFlags,
        PipelineDynamicStateCreateInfo,
        PipelineDynamicStateCreateFlags,
        PipelineLayoutCreateInfo,
        PipelineLayoutCreateFlags,
        PipelineLayout,
        PipelineViewportStateCreateInfo,
        PipelineViewportStateCreateFlags,
        Rect2D,
        Offset2D,
        Viewport,
        Pipeline,
        GraphicsPipelineCreateInfo,
        PipelineCreateFlags,
        PipelineCache,

        Image,
        ImageViewCreateInfo,
        ImageViewCreateFlags,
        ComponentSwizzle,
        ImageAspectFlags,
        ImageViewType,
        ImageView,
        ComponentMapping,
        ImageSubresourceRange,
        FramebufferCreateInfo,
        FramebufferCreateFlags,
        Framebuffer,

        CommandPoolCreateInfo,
        CommandPoolCreateFlags,
        CommandPool,
        CommandBufferAllocateInfo,
        CommandBufferLevel,
        CommandBuffer,
        CommandBufferBeginInfo,
        CommandBufferUsageFlags,
        RenderPassBeginInfo,
        ClearValue,
        ClearColorValue,
        SubpassContents,

        SemaphoreCreateInfo,
        SemaphoreCreateFlags,
        FenceCreateInfo,
        FenceCreateFlags,
        SubmitInfo,
        PipelineStageFlags,
        SubpassDependency,
        PresentInfoKHR,
        Semaphore,
        Fence,
    },
    extensions::ext::DebugUtils,
    Entry,
    LoadingError,
    Instance,
    Device,
    extensions::khr::{
        Surface,
        Swapchain
    }
};
use winit::{
    window::Window,
    event_loop::{
        EventLoop,
        ControlFlow
    },
    event::{
        Event,
        WindowEvent
    },
    error::OsError
};
use std::{
    ptr::{
        null,
        null_mut
    },
    ffi::{
        c_void,
        CStr
    },
    num::TryFromIntError
};

fn main() -> AppResult<()> {

    let entry = unsafe { Entry::load()? };
    let event_loop = EventLoop::new();
    let window = Window::new(&event_loop)?;
    let instance = create_instance(&entry, &window)?;
    #[cfg(debug_assertions)]
    let debug_utils = DebugUtils::new(&entry, &instance);
    #[cfg(debug_assertions)]
    let debug_messenger = create_debug_utils_messenger(&debug_utils)?;
    let surface = Surface::new(&entry, &instance);
    let surface_khr = unsafe { ash_window::create_surface(&entry, &instance, &window, None)? };
    let physical_device = get_physical_device(&instance)?;
    let queue_index = get_queue_family_index(&instance, &surface, surface_khr, physical_device)?;
    let device = create_device(&instance, physical_device, queue_index)?;
    let queue = unsafe { device.get_device_queue(queue_index, 0) };
    let command_pool = create_command_pool(&device, queue_index)?;
    let swapchain = Swapchain::new(&instance, &device);

    let capabilities = unsafe {surface.get_physical_device_surface_capabilities(physical_device, surface_khr)? };
    let mut format = get_format(&surface, surface_khr, physical_device)?;
    let mut extent = get_extent(&window, &capabilities);
    let mut swapchain_khr = create_swapchain(&swapchain, surface_khr, &capabilities, format, extent, SwapchainKHR::null())?;
    let images = unsafe { swapchain.get_swapchain_images(swapchain_khr)? };
    let mut image_views = create_image_views(&device, format.format, &images)?;
    let mut render_pass = create_render_pass(&device, format.format)?;
    let mut framebuffers = create_framebuffers(&device, &image_views, extent, render_pass)?;

    let command_buffers = create_command_buffers(&device, command_pool, images.len().try_into()?)?;

    let vertex_module = create_shader_module(&device, include_bytes!("shaders/voxel_vertex.spv"))?;
    let fragment_module = create_shader_module(&device, include_bytes!("shaders/voxel_fragment.spv"))?;
    let layout = create_pipeline_layout(&device)?;
    let pipeline = create_pipeline(&device, render_pass, extent, vertex_module, fragment_module, layout)?;

    record_command_buffers(&device, command_buffers.clone(), framebuffers.clone(), render_pass, extent, pipeline)?;
    let image_available = create_semaphore(&device)?;
    let render_finished = create_semaphore(&device)?;
    let in_flight = create_fence(&device)?;

    unsafe {
        device.destroy_pipeline_layout(layout, None);
        device.destroy_shader_module(vertex_module, None);
        device.destroy_shader_module(fragment_module, None);
    }

    let mut window_extent = extent;
    let mut running = true;
    let mut recreate_swapchain = false;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        match event {
            Event::MainEventsCleared => {
                let res: AppResult<()> = try {

                    if window_extent.width == 0 || window_extent.height == 0 || !running {
                        return
                    }

                    unsafe { device.wait_for_fences(&[in_flight], true, u64::MAX)? };

                    let image_index = unsafe { swapchain.acquire_next_image(swapchain_khr, u64::MAX, image_available, Fence::null())? };
                    if image_index.1 {
                        recreate_swapchain = true;
                    }
                    let image_index = image_index.0;

                    unsafe { device.reset_fences(&[in_flight])? };
                        
                    let submit_info = SubmitInfo {
                        s_type: StructureType::SUBMIT_INFO,
                        p_next: null(),
                        wait_semaphore_count: 1,
                        p_wait_semaphores: &image_available,
                        p_wait_dst_stage_mask: &PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        command_buffer_count: 1,
                        p_command_buffers: &command_buffers[image_index as usize],
                        signal_semaphore_count: 1,
                        p_signal_semaphores: &render_finished
                    };

                    unsafe { device.queue_submit(queue, &[submit_info], in_flight)? };

                    let present_info = PresentInfoKHR {
                        s_type: StructureType::PRESENT_INFO_KHR,
                        p_next: null(),
                        wait_semaphore_count: 1,
                        p_wait_semaphores: &render_finished,
                        swapchain_count: 1,
                        p_swapchains: &swapchain_khr,
                        p_image_indices: &image_index,
                        p_results: null_mut()
                    };

                    if unsafe { swapchain.queue_present(queue, &present_info)? } {
                        recreate_swapchain = true;
                    }

                };
                if recreate_swapchain {
                    let res: AppResult<()> = try {
                        unsafe {
                            let capabilities = surface.get_physical_device_surface_capabilities(physical_device, surface_khr)?;
                            extent = get_extent(&window, &capabilities);
                            if extent.width == 0 || extent.height == 0 { return }

                            device.device_wait_idle()?;
                            framebuffers.iter().for_each(|framebuffer| device.destroy_framebuffer(*framebuffer, None));
                            image_views.iter().for_each(|image_view| device.destroy_image_view(*image_view, None));
                            let new_format = get_format(&surface, surface_khr, physical_device)?;
                            if new_format != format {
                                format = new_format;
                                device.destroy_render_pass(render_pass, None);
                                render_pass = create_render_pass(&device, format.format)?;
                            }
                            let new_swapchain_khr = create_swapchain(&swapchain, surface_khr, &capabilities, format, extent, swapchain_khr)?;
                            swapchain.destroy_swapchain(swapchain_khr, None);
                            swapchain_khr = new_swapchain_khr;
                            let images = swapchain.get_swapchain_images(swapchain_khr)?;
                            image_views = create_image_views(&device, format.format, &images)?;
                            framebuffers = create_framebuffers(&device, &image_views, extent, render_pass)?;
                            record_command_buffers(&device, command_buffers.clone(), framebuffers.clone(), render_pass, extent, pipeline)?;
                            window_extent = extent;
                            recreate_swapchain = false;
                        } 
                    };
                    res.unwrap();
                }
            },
            Event::WindowEvent {
                event: WindowEvent::Resized(new_size),
                ..
            } => {
                recreate_swapchain = true;
                window_extent = Extent2D { width: new_size.width, height: new_size.height }
            },
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                unsafe {
                    device.device_wait_idle().ok();
                    device.destroy_semaphore(image_available, None);
                    device.destroy_semaphore(render_finished, None);
                    device.destroy_fence(in_flight, None);
                    device.destroy_pipeline(pipeline, None);

                    framebuffers.iter().for_each(|framebuffer| device.destroy_framebuffer(*framebuffer, None));
                    device.destroy_render_pass(render_pass, None);
                    image_views.iter().for_each(|image_view| device.destroy_image_view(*image_view, None));
                    swapchain.destroy_swapchain(swapchain_khr, None);

                    device.destroy_command_pool(command_pool, None);
                    device.destroy_device(None);
                    surface.destroy_surface(surface_khr, None);
                    #[cfg(debug_assertions)]
                    debug_utils.destroy_debug_utils_messenger(debug_messenger, None);
                    instance.destroy_instance(None);

                }
                *control_flow = ControlFlow::Exit;
                running = false;
            },
            _ => ()
        }
    });
}

fn create_fence(device: &Device) -> AppResult<Fence> {

    let create_info = FenceCreateInfo {
        s_type: StructureType::FENCE_CREATE_INFO,
        p_next: null(),
        flags: FenceCreateFlags::SIGNALED,
    };

    unsafe { Ok(device.create_fence(&create_info, None)?) }

}

fn create_semaphore(device: &Device) -> AppResult<Semaphore> {

    let create_info = SemaphoreCreateInfo {
        s_type: StructureType::SEMAPHORE_CREATE_INFO,
        p_next: null(),
        flags: SemaphoreCreateFlags::empty(),
    };

    unsafe { Ok(device.create_semaphore(&create_info, None)?) }
}

fn record_command_buffers(device: &Device, mut command_buffers: Vec<CommandBuffer>, mut frame_buffers: Vec<Framebuffer>, render_pass: RenderPass, extent: Extent2D, pipeline: Pipeline) -> AppResult<()> {

    for (command_buffer, framebuffer) in command_buffers.drain(..).zip(frame_buffers.drain(..)).collect::<Vec<(CommandBuffer, Framebuffer)>>() {
        
        let begin_info = CommandBufferBeginInfo {
            s_type: StructureType::COMMAND_BUFFER_BEGIN_INFO,
            p_next: null(),
            flags: CommandBufferUsageFlags::empty(),
            p_inheritance_info: null()
        };

        unsafe { device.begin_command_buffer(command_buffer, &begin_info)? };

        let clear_value = ClearValue { color: ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } };

        let render_begin_info = RenderPassBeginInfo {
            s_type: StructureType::RENDER_PASS_BEGIN_INFO,
            p_next: null(),
            render_pass: render_pass,
            framebuffer: framebuffer,
            render_area: Rect2D {
                offset: Offset2D {
                    x: 0,
                    y: 0
                },
                extent: extent
            },
            clear_value_count: 1,
            p_clear_values: &clear_value
        };

        let viewport = Viewport {
            x: 0.0,
            y: 0.0,
            height: extent.height as f32,
            width: extent.width as f32,
            min_depth: 0.0,
            max_depth: 1.0
        };

        let scissors = Rect2D {
            offset: Offset2D {
                x: 0,
                y: 0
            },
            extent
        };

        unsafe {
            device.cmd_begin_render_pass(command_buffer, &render_begin_info, SubpassContents::INLINE);
            device.cmd_bind_pipeline(command_buffer, PipelineBindPoint::GRAPHICS, pipeline);
            device.cmd_set_viewport(command_buffer, 0, &[viewport]);
            device.cmd_set_scissor(command_buffer, 0, &[scissors]);
            device.cmd_draw(command_buffer, 3, 1, 0, 0);
            device.cmd_end_render_pass(command_buffer);
            device.end_command_buffer(command_buffer)?;
        }
    }
    Ok(())
}

fn create_command_buffers(device: &Device, command_pool: CommandPool, count: u32) -> AppResult<Vec<CommandBuffer>> {

    let alloc_info = CommandBufferAllocateInfo {
        s_type: StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
        p_next: null(),
        command_pool: command_pool,
        level: CommandBufferLevel::PRIMARY,
        command_buffer_count: count
    };

    unsafe { Ok(device.allocate_command_buffers(&alloc_info)?) }
}

fn create_command_pool(device: &Device, queue_index: u32) -> AppResult<CommandPool> {

    let create_info = CommandPoolCreateInfo {
        s_type: StructureType::COMMAND_POOL_CREATE_INFO,
        p_next: null(),

        //change this later on
        flags: CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        queue_family_index: queue_index,
    };

    unsafe { Ok(device.create_command_pool(&create_info, None)?) }
}

fn create_image_views(device: &Device, format: Format, images: &Vec<Image>) -> AppResult<Vec<ImageView>> {
    images.iter().map(|image| {

        let create_info = ImageViewCreateInfo {
            s_type: StructureType::IMAGE_VIEW_CREATE_INFO,
            p_next: null(),
            flags: ImageViewCreateFlags::empty(),
            image: *image,
            view_type: ImageViewType::TYPE_2D,
            format: format,
            components: ComponentMapping {
                r: ComponentSwizzle::IDENTITY,
                g: ComponentSwizzle::IDENTITY,
                b: ComponentSwizzle::IDENTITY,
                a: ComponentSwizzle::IDENTITY,
            },
            subresource_range: ImageSubresourceRange {
                aspect_mask: ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1
            }
        };

        unsafe { device.create_image_view(&create_info, None) }

    }).collect::<Result<Vec<ImageView>,_>>().map_err(|err| From::from(err))
}

fn create_framebuffers(device: &Device, images: &Vec<ImageView>, extent: Extent2D, render_pass: RenderPass) -> AppResult<Vec<Framebuffer>> {
    images.iter().map(|image| {

        let create_info = FramebufferCreateInfo {
            s_type: StructureType::FRAMEBUFFER_CREATE_INFO,
            p_next: null(),
            flags: FramebufferCreateFlags::empty(),
            render_pass: render_pass,
            attachment_count: 1,
            p_attachments: image,
            width: extent.width,
            height: extent.height,
            layers: 1
        };
        
        unsafe { device.create_framebuffer(&create_info, None) }
    }).collect::<Result<Vec<Framebuffer>, _>>().map_err(|err| From::from(err))
}

fn create_pipeline(device: &Device, render_pass: RenderPass, extent: Extent2D, vertex_module: ShaderModule, fragment_module: ShaderModule, layout: PipelineLayout) -> AppResult<Pipeline> {

    let vertex_stage_info = PipelineShaderStageCreateInfo {
        s_type: StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
        p_next: null(),
        flags: PipelineShaderStageCreateFlags::empty(),
        stage: ShaderStageFlags::VERTEX,
        module: vertex_module,
        p_name: b"main\0".as_ptr() as *const i8,
        p_specialization_info: null()
    };

    let fragment_stage_info = PipelineShaderStageCreateInfo {
        s_type: StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
        p_next: null(),
        flags: PipelineShaderStageCreateFlags::empty(),
        stage: ShaderStageFlags::FRAGMENT,
        module: fragment_module,
        p_name: b"main\0".as_ptr() as *const i8,
        p_specialization_info: null()
    };

    let stage_infos = [vertex_stage_info, fragment_stage_info];

    let vertex_input_info = PipelineVertexInputStateCreateInfo {
        s_type: StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        p_next: null(),
        flags: PipelineVertexInputStateCreateFlags::empty(),
        vertex_binding_description_count: 0,
        p_vertex_binding_descriptions: null(),
        vertex_attribute_description_count: 0,
        p_vertex_attribute_descriptions: null(),
    };

    let input_assembly = PipelineInputAssemblyStateCreateInfo {
        s_type: StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        p_next: null(),
        flags: PipelineInputAssemblyStateCreateFlags::empty(),
        topology: PrimitiveTopology::TRIANGLE_LIST,
        primitive_restart_enable: FALSE
    };

    let rasterizer = PipelineRasterizationStateCreateInfo {
        s_type: StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        p_next: null(),
        flags: PipelineRasterizationStateCreateFlags::empty(),
        depth_clamp_enable: FALSE,
        rasterizer_discard_enable: FALSE,
        polygon_mode: PolygonMode::FILL,
        line_width: 1.0,
        cull_mode: CullModeFlags::BACK,
        front_face: FrontFace::CLOCKWISE,
        depth_bias_enable: FALSE,
        depth_bias_constant_factor: 0.0,
        depth_bias_clamp: 0.0,
        depth_bias_slope_factor: 0.0
    };

    let multisampling = PipelineMultisampleStateCreateInfo {
        s_type: StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        p_next: null(),
        flags: PipelineMultisampleStateCreateFlags::empty(),
        sample_shading_enable: FALSE,
        rasterization_samples: SampleCountFlags::TYPE_1,
        min_sample_shading: 1.0,
        p_sample_mask: null(),
        alpha_to_coverage_enable: FALSE,
        alpha_to_one_enable: FALSE
    };

    let color_blend_attachment = PipelineColorBlendAttachmentState {
        color_write_mask: ColorComponentFlags::R | ColorComponentFlags::G | ColorComponentFlags::B | ColorComponentFlags::A,
        blend_enable: FALSE,
        src_color_blend_factor: BlendFactor::ONE,
        dst_color_blend_factor: BlendFactor::ZERO,
        color_blend_op: BlendOp::ADD,
        src_alpha_blend_factor: BlendFactor::ONE,
        dst_alpha_blend_factor: BlendFactor::ZERO,
        alpha_blend_op: BlendOp::ADD
    };

    let color_blending = PipelineColorBlendStateCreateInfo {
        s_type: StructureType::PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        p_next: null(),
        flags: PipelineColorBlendStateCreateFlags::empty(),
        logic_op_enable: FALSE,
        logic_op: LogicOp::COPY,
        attachment_count: 1,
        p_attachments: &color_blend_attachment,
        blend_constants: [0.0, 0.0, 0.0, 0.0]
    };

    let viewport = Viewport {
        x: 0.0,
        y: 0.0,
        width: extent.width as f32,
        height: extent.height as f32,
        min_depth: 0.0,
        max_depth: 1.0
    };

    let scissor = Rect2D {
        offset: Offset2D {
            x: 0,
            y: 0
        },
        extent: extent
    };

    let viewport_state = PipelineViewportStateCreateInfo {
        s_type: StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        p_next: null(),
        flags: PipelineViewportStateCreateFlags::empty(),
        viewport_count: 1,
        p_viewports: &viewport,
        scissor_count: 1,
        p_scissors: &scissor
    };

    let dynamic_states = [DynamicState::VIEWPORT, DynamicState::SCISSOR];

    let dynamic_state = PipelineDynamicStateCreateInfo {
        s_type: StructureType::PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        p_next: null(),
        flags: PipelineDynamicStateCreateFlags::empty(),
        dynamic_state_count: dynamic_states.len().try_into()?,
        p_dynamic_states: dynamic_states.as_ptr()
    };

    let create_info = GraphicsPipelineCreateInfo {
        s_type: StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
        p_next: null(),
        flags: PipelineCreateFlags::empty(),
        stage_count: stage_infos.len().try_into()?,
        p_stages: stage_infos.as_ptr(),
        p_vertex_input_state: &vertex_input_info,
        p_input_assembly_state: &input_assembly,
        p_viewport_state: &viewport_state,
        p_rasterization_state: &rasterizer,
        p_multisample_state: &multisampling,
        p_depth_stencil_state: null(),
        p_color_blend_state: &color_blending,
        p_dynamic_state: &dynamic_state,
        p_tessellation_state: null(),
        layout: layout,
        render_pass: render_pass,
        subpass: 0,
        base_pipeline_handle: Pipeline::null(),
        base_pipeline_index: -1
    };

    unsafe { Ok(device.create_graphics_pipelines(PipelineCache::null(), &[create_info], None).map_err(|err| err.1)?.pop().ok_or(Error::GpuNotFound)?) }
}

fn create_pipeline_layout(device: &Device) -> AppResult<PipelineLayout> {

    let create_info = PipelineLayoutCreateInfo {
        s_type: StructureType::PIPELINE_LAYOUT_CREATE_INFO,
        p_next: null(),
        flags: PipelineLayoutCreateFlags::empty(),
        set_layout_count: 0,
        p_set_layouts: null(),
        push_constant_range_count: 0,
        p_push_constant_ranges: null()
    };

    unsafe { Ok(device.create_pipeline_layout(&create_info, None)?) }
}

fn create_shader_module(device: &Device, code: &[u8]) -> AppResult<ShaderModule> {

    let create_info = ShaderModuleCreateInfo {
        s_type: StructureType::SHADER_MODULE_CREATE_INFO,
        p_next: null(),
        flags: ShaderModuleCreateFlags::empty(),
        code_size: code.len(),
        p_code: code.as_ptr() as *const u32
    };

    unsafe { Ok(device.create_shader_module(&create_info, None)?) }
}

fn create_render_pass(device: &Device, format: Format) -> AppResult<RenderPass> {

    let color_attachment = AttachmentDescription {
        flags: AttachmentDescriptionFlags::empty(),
        format: format,
        samples: SampleCountFlags::TYPE_1,
        load_op: AttachmentLoadOp::CLEAR,
        store_op: AttachmentStoreOp::STORE,
        stencil_load_op: AttachmentLoadOp::DONT_CARE,
        stencil_store_op: AttachmentStoreOp::DONT_CARE,
        initial_layout: ImageLayout::UNDEFINED,
        final_layout: ImageLayout::PRESENT_SRC_KHR
    };

    let color_attachment_ref = AttachmentReference {
        attachment: 0,
        layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    };

    let subpass = SubpassDescription {
        flags: SubpassDescriptionFlags::empty(),
        pipeline_bind_point: PipelineBindPoint::GRAPHICS,
        color_attachment_count: 1,
        p_color_attachments: &color_attachment_ref,
        input_attachment_count: 0,
        p_input_attachments: null(),
        p_resolve_attachments: null(),
        preserve_attachment_count: 0,
        p_preserve_attachments: null(),
        p_depth_stencil_attachment: null()
    };

    let dependency = SubpassDependency {
        src_subpass: ash::vk::SUBPASS_EXTERNAL,
        dst_subpass: 0,
        src_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        src_access_mask: AccessFlags::empty(),
        dst_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        dst_access_mask: AccessFlags::COLOR_ATTACHMENT_WRITE,
        dependency_flags: DependencyFlags::empty()
    };

    let create_info = RenderPassCreateInfo {
        s_type: StructureType::RENDER_PASS_CREATE_INFO,
        p_next: null(),
        flags: RenderPassCreateFlags::empty(),
        attachment_count: 1,
        p_attachments: &color_attachment,
        subpass_count: 1,
        p_subpasses: &subpass,
        dependency_count: 1,
        p_dependencies: &dependency
    };

    unsafe { Ok(device.create_render_pass(&create_info, None)?) }
}

fn create_swapchain(swapchain: &Swapchain, surface_khr: SurfaceKHR, caps: &SurfaceCapabilitiesKHR, format: SurfaceFormatKHR, extent: Extent2D, old_swapchain: SwapchainKHR) -> AppResult<SwapchainKHR> {

    let create_info = SwapchainCreateInfoKHR {
        s_type: StructureType::SWAPCHAIN_CREATE_INFO_KHR,
        p_next: null(),
        flags: SwapchainCreateFlagsKHR::empty(),
        surface: surface_khr,
        min_image_count: caps.min_image_count.saturating_add(1)
            .min(caps.max_image_count.wrapping_sub(1).saturating_add(1)),
        image_format: format.format,
        image_color_space: format.color_space,
        image_extent: extent,
        image_array_layers: 1,
        image_usage: ImageUsageFlags::COLOR_ATTACHMENT,
        image_sharing_mode: SharingMode::EXCLUSIVE,
        queue_family_index_count: 0,
        p_queue_family_indices: null(),
        pre_transform: caps.current_transform,
        composite_alpha: CompositeAlphaFlagsKHR::OPAQUE,
        present_mode: PresentModeKHR::FIFO,
        clipped: TRUE,
        old_swapchain: old_swapchain
    };

    unsafe { Ok(swapchain.create_swapchain(&create_info, None)?) }
}

fn get_extent(window: &Window, caps: &SurfaceCapabilitiesKHR) -> Extent2D {
    if caps.current_extent.width != u32::MAX { caps.current_extent }
    else {
        let size = window.inner_size();
        Extent2D {
            width: size.width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
            height: size.height.clamp(caps.min_image_extent.height, caps.max_image_extent.height)
        }
    }
}

fn get_format(surface: &Surface, surface_khr: SurfaceKHR, device: PhysicalDevice) -> AppResult<SurfaceFormatKHR> {
    Ok(unsafe { surface.get_physical_device_surface_formats(device, surface_khr)? }
        .into_iter().enumerate().rev()
        .find_map(|(index, format)| if (format.format == Format::B8G8R8A8_SRGB &&
                                 format.color_space == ColorSpaceKHR::SRGB_NONLINEAR)
                                 || index == 0 { Some(format) } else { None })
        .ok_or(Error::GpuNotFound)?)
}

fn create_device(instance: &Instance, physical_device: PhysicalDevice, queue_family_index: u32) -> AppResult<Device> {

    let extensions = &[b"VK_KHR_swapchain\0".as_ptr() as *const i8];

    let queue_create_info = DeviceQueueCreateInfo {
        s_type: StructureType::DEVICE_QUEUE_CREATE_INFO,
        p_next: null(),
        flags: DeviceQueueCreateFlags::empty(),
        queue_family_index: queue_family_index,
        queue_count: 1,
        p_queue_priorities: &1f32
    };

    let create_info = DeviceCreateInfo {
        s_type: StructureType::DEVICE_CREATE_INFO,
        p_next: null(),
        flags: DeviceCreateFlags::empty(),
        queue_create_info_count: 1,
        p_queue_create_infos: &queue_create_info,
        p_enabled_features: null(),
        enabled_extension_count: extensions.len().try_into()?,
        pp_enabled_extension_names: extensions.as_ptr(),
        enabled_layer_count: 0,
        pp_enabled_layer_names: null()
    };

    unsafe { Ok(instance.create_device(physical_device, &create_info, None)?) }
}

fn get_physical_device(instance: &Instance) -> AppResult<PhysicalDevice> {
    unsafe { Ok(instance.enumerate_physical_devices()?.pop().ok_or(Error::GpuNotFound)?) }
}

fn get_queue_family_index(instance: &Instance, surface: &Surface, surface_khr: SurfaceKHR, device: PhysicalDevice) -> AppResult<u32> {
    Ok( unsafe { instance.get_physical_device_queue_family_properties(device) }
        .iter().enumerate()
        .find_map(|(index, queue)| {
            match index.try_into() {
                Ok(index) => if queue.queue_flags.contains(QueueFlags::GRAPHICS) &&
                unsafe {
                    surface.get_physical_device_surface_support(device, index, surface_khr)
                    .unwrap_or(false) 
                } { Some(index) } else { None },
                Err(_) => None
            }
        }).ok_or(Error::GpuNotFound)?)
}

#[cfg(debug_assertions)]
fn create_debug_utils_messenger(debug_utils: &DebugUtils) -> AppResult<DebugUtilsMessengerEXT> {

    extern "system" fn debug_callback(_: DebugUtilsMessageSeverityFlagsEXT,
                                      _: DebugUtilsMessageTypeFlagsEXT,
                                      data: *const DebugUtilsMessengerCallbackDataEXT,
                                      _: *mut c_void) -> u32 {

        println!("{}", unsafe {
            CStr::from_ptr((*data).p_message).to_str()
                .unwrap_or("Failed to decode validation message!")
        });
        FALSE
    }

    let create_info = DebugUtilsMessengerCreateInfoEXT {
        s_type: StructureType::DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        p_next: null(),
        flags: DebugUtilsMessengerCreateFlagsEXT::empty(),
        message_severity: DebugUtilsMessageSeverityFlagsEXT::ERROR |
            DebugUtilsMessageSeverityFlagsEXT::WARNING |
            DebugUtilsMessageSeverityFlagsEXT::VERBOSE |
            DebugUtilsMessageSeverityFlagsEXT::INFO,
        message_type: DebugUtilsMessageTypeFlagsEXT::GENERAL |
            DebugUtilsMessageTypeFlagsEXT::PERFORMANCE |
            DebugUtilsMessageTypeFlagsEXT::VALIDATION,
        pfn_user_callback: Some(debug_callback),
        p_user_data: null_mut()
    };

    unsafe { Ok(debug_utils.create_debug_utils_messenger(&create_info, None)?) }
}

pub fn create_instance(entry: &Entry, window: &Window) -> AppResult<Instance> {

    #[cfg(debug_assertions)]
    let extensions = [b"VK_EXT_debug_utils\0".as_ptr() as *const i8];
    #[cfg(not(debug_assertions))]
    let extensions = [];

    let extensions = [&extensions, ash_window::enumerate_required_extensions(window)?].concat();

    #[cfg(debug_assertions)]
    let layers = [b"VK_LAYER_KHRONOS_validation\0".as_ptr() as *const i8];
    #[cfg(not(debug_assertions))]
    let layers = [];

    let application_info = ApplicationInfo {
        s_type: StructureType::APPLICATION_INFO,
        p_next: null(),
        p_application_name: b"unnamed\0".as_ptr() as *const i8,
        application_version: make_api_version(0, 0, 1, 0),
        p_engine_name: b"unnamed\0".as_ptr() as *const i8,
        engine_version: make_api_version(0, 0, 1, 0),
        api_version: make_api_version(0, 1, 3, 0)
    };

    let create_info = InstanceCreateInfo {
        s_type: StructureType::INSTANCE_CREATE_INFO,
        p_next: null(),
        flags: InstanceCreateFlags::empty(),
        p_application_info: &application_info,
        enabled_extension_count: extensions.len().try_into()?,
        pp_enabled_extension_names: extensions.as_ptr(),
        enabled_layer_count: layers.len().try_into()?,
        pp_enabled_layer_names: layers.as_ptr()
    };

    unsafe { Ok(entry.create_instance(&create_info, None)?) }
}


pub type AppResult<T> = Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    VkResult(VkResult),
    LoadingError(LoadingError),
    TryFromIntError(TryFromIntError),
    OsError(OsError),
    GpuNotFound
}
impl From<VkResult> for Error {
    fn from(err: VkResult) -> Error {
        Error::VkResult(err)
    }
}
impl From<LoadingError> for Error {
    fn from(err: LoadingError) -> Error {
        Error::LoadingError(err)
    }
}
impl From<TryFromIntError> for Error {
    fn from(err: TryFromIntError) -> Error {
        Error::TryFromIntError(err)
    }
}
impl From<OsError> for Error {
    fn from(err: OsError) -> Error {
        Error::OsError(err)
    }
}
