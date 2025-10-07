use triangulate::triangulate::wgpu_impl;

// ...

                group.bench_function("triangulate-wgpu", |b| {
                    b.iter(|| {
                        _ = wgpu_impl::triangulate(&step_file);
                    });
                });